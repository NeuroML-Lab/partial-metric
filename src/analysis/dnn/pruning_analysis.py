import os
import copy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Optional

import torch
import torch.nn as nn

from src.analysis.dnn.utils import (
    compute_accuracy,
    imagenet_dataloader,
    load_weights,
    get_layer_names,
    get_activations,
)
from src.metrics.alignment.partial_wasserstein import UnbalancedSoftMatch


def get_module_by_name(model: torch.nn.Module, layer_name: str, fuzzy: bool = True):
    """Return the module instance matching layer_name (exact or fuzzy)."""
    named = dict(model.named_modules())
    if layer_name in named:
        return named[layer_name]
    if fuzzy:
        # prefer shortest candidate that endswith or contains layer_name
        candidates = [
            n for n in named.keys() if n.endswith(layer_name) or layer_name in n
        ]
        if candidates:
            best = sorted(candidates, key=lambda s: (len(s), s))[0]
            return named[best]
    raise KeyError(
        f"Module '{layer_name}' not found. Candidates: {list(named.keys())[:30]}..."
    )


def find_following_batchnorm(
    model: nn.Module, target_mod: nn.Module
) -> Optional[nn.Module]:
    """
    Find the first BatchNorm module that occurs after `target_mod` in model.named_modules() order.
    Returns the module or None if not found.
    """
    named = list(model.named_modules())  # list of (name, module)
    # find index of target module by identity
    idx = None
    for i, (_, mod) in enumerate(named):
        if mod is target_mod:
            idx = i
            break
    if idx is None:
        return None
    # search forward for first BatchNorm module
    for _, mod in named[idx + 1 :]:
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return mod
    return None


def register_zero_activation_hook(model: nn.Module, layer_name: str, ch_idx: int):
    """
    Register a hook that zeros channel ch_idx of the
    activation returned by the module named layer_name.
    """
    mod = get_module_by_name(model, layer_name)

    def hook_fn(module, inp, out):
        # mutable replacement: zero-out that output channel
        # handle 2D conv maps (B,C,H,W) and dense outputs (B,C)
        out = out.clone()
        if out.dim() == 4:
            out[:, ch_idx, :, :] = 0.0
        elif out.dim() == 2:
            out[:, ch_idx] = 0.0
        else:
            # try to zero last dims if channel-like
            out[:, ch_idx, ...] = 0.0
        return out

    handle = mod.register_forward_hook(hook_fn)
    bn_mod = find_following_batchnorm(model, mod)
    old_weight = None
    old_bias = None
    if bn_mod is not None:
        # Only proceed if BN has affine parameters
        if (
            getattr(bn_mod, "weight", None) is not None
            and getattr(bn_mod, "bias", None) is not None
        ):
            # save original values (CPU clones) so we can restore later
            old_weight = bn_mod.weight.detach().cpu().clone()
            old_bias = bn_mod.bias.detach().cpu().clone()
            # zero the specified channel's affine params in-place (on BN's device)
            with torch.no_grad():
                device = bn_mod.weight.device
                bn_mod.weight.data = bn_mod.weight.data.to(device)
                bn_mod.bias.data = bn_mod.bias.data.to(device)
                try:
                    bn_mod.weight.data[ch_idx] = 0.0
                    bn_mod.bias.data[ch_idx] = 0.0
                except Exception as e:
                    # If indexing fails (e.g., ch_idx out of range) we quietly skip
                    print(
                        f"[register_zero_activation_hook] warning: could not zero BN channel {ch_idx}: {e}"
                    )
        else:
            # BN has no affine params (affine=False) — nothing to zero
            bn_mod = None

    return {
        "hook_handle": handle,
        "bn_module": bn_mod,
        "old_bn_weight": old_weight,
        "old_bn_bias": old_bias,
    }
    # return handle


def restore_batchnorm_params(
    bn_module: nn.Module,
    old_weight: Optional[torch.Tensor],
    old_bias: Optional[torch.Tensor],
):
    """
    Restore saved BatchNorm affine parameters into bn_module.
    `old_weight` and `old_bias` should be CPU tensors (or None).
    """
    if bn_module is None or old_weight is None or old_bias is None:
        return
    with torch.no_grad():
        device = bn_module.weight.device
        bn_module.weight.data = old_weight.to(device)
        bn_module.bias.data = old_bias.to(device)


def prune_network(
    layer_names: List[str],
    results_dir: str,
    device_id: int,
    model1: nn.Module,
    model2: nn.Module,
    dloader: torch.utils.data.DataLoader,
    args,
    use_hook_ablation: bool = False,
):
    """
    prune a network and measure accuracy using 3 pruning strategies:
        a. random: randomly choose same number of units as predicted to prune by transport plan
        b. mean_abs: prune units with lowest mean absolute activation (over spatial dims)
        c. unbalanced_reg: prune units with near-zero outgoing mass in the transport plan

    parameters:
        layer_names (List[str]): list of module names to prune
        results_dir (str): directory to save per-layer results
        device_id (str): GPU id
        model1, model2 (nn.Module): trained PyTorch model instances
        dloader (torch.utils.data.DataLoader): dataloader used to compute feature maps and accuracy
        args: namespace for args.results_dir used for saving results
        use_hook_ablation (bool): if True, use a forward-hook ablation
    """

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    mreg = np.arange(0.0, 1.0, 0.1)
    thresh = 1e-5  # transport weight where we consider no "connection"
    pruning_strategies = ["random", "mean_abs", "unbalanced_reg"]

    # loop over all layers
    for layer_to_prune in layer_names:
        fpath = os.path.join(results_dir, f"{layer_to_prune.replace('/', '_')}.pkl")
        print(f"\npruning layer: {layer_to_prune} -> saving to {fpath}")

        if os.path.exists(fpath):
            print(f"results exist, skipping to next layer")
            continue

        # dictionary to store model accuracies at corresponding mreg values
        acc_dict = {
            float(m): {
                "mean_abs": None,
                "unbalanced_reg": None,
                "random": None,
                "mean_abs_idx": None,
                "unbalanced_sm_idx": None,
                "unbalanced_transform": None,
            }
            for m in mreg
        }

        # get activations for both models
        activations_model1 = get_activations(
            model=model1,
            dloader=dloader,
            layer_name=layer_to_prune,
            device_id=device_id,
        )
        activations_model2 = get_activations(
            model=model2,
            dloader=dloader,
            layer_name=layer_to_prune,
            device_id=device_id,
        )

        # now, loop over all mreg values
        for m in tqdm(mreg):
            # fit partial soft-permutation matrix and compute alignment
            unbalanced_sm = UnbalancedSoftMatch(mass_reg=float(m))
            _ = unbalanced_sm.fit_kfold_score(
                activations_model1.cpu(), activations_model2.cpu()
            )
            transport_plan = unbalanced_sm.transform
            # get best (mutual) matches
            neuron_idx = torch.where(transport_plan.sum(axis=1) < thresh)[0]
            # src_best = torch.argmax(transport_plan, dim=1)
            # target_best = torch.argmax(transport_plan, dim=0)
            # neuron_idx = []
            # for i in range(transport_plan.shape[0]):
            #    j = int(src_best[i])
            #    if transport_plan[i, j] > 1e-3 and target_best[j] == i:
            #        neuron_idx.append(i)

            num_units_to_prune = len(neuron_idx)
            # populate dictionary with transport plan and dropped neuron indices
            acc_dict[float(m)]["unbalanced_transform"] = transport_plan.numpy()
            acc_dict[float(m)]["unbalanced_sm_idx"] = neuron_idx

            for strategy in pruning_strategies:
                # deep-copy model1 (safely): move copy to CPU first
                # to avoid GPU state copy issues
                model_copy = copy.deepcopy(model1).cpu()
                model_copy = model_copy.to(device).eval()
                # get submodule to prune
                submodule = dict(model_copy.named_modules()).get(layer_to_prune)

                if use_hook_ablation:
                    handles = []
                    if strategy == "mean_abs":
                        # compute mean absolute activations over all stimuli
                        mean_abs = torch.mean(torch.abs(activations_model1), dim=0)
                        bottom_channel_idx = torch.argsort(mean_abs)[
                            :num_units_to_prune
                        ]
                        acc_dict[float(m)]["mean_abs_idx"] = (
                            bottom_channel_idx.cpu().numpy().tolist()
                        )
                        targets = bottom_channel_idx.tolist()
                    elif strategy == "unbalanced_reg":
                        targets = neuron_idx  # .tolist()
                    elif strategy == "random":
                        out_channels = submodule.weight.shape[0]
                        random_indices = torch.randperm(out_channels)[
                            :num_units_to_prune
                        ]
                        targets = random_indices.tolist()
                    else:
                        raise ValueError(f"invalid pruning strategy")

                    # register hooks on the module we want to 0 out
                    for neuron in targets:
                        handles.append(
                            register_zero_activation_hook(
                                model_copy, layer_to_prune, int(neuron)
                            )
                        )

                    # compute accuracy while the 0-hook is active
                    model_acc = compute_accuracy(
                        model=model_copy, dloader=dloader, device_id=device_id
                    )
                    # remove hooks once this is done
                    for h in handles:
                        h["hook_handle"].remove()
                        restore_batchnorm_params(
                            h["bn_module"], h["old_bn_weight"], h["old_bn_bias"]
                        )

                print(
                    f"m={m:.2f} strategy={strategy} pruned={num_units_to_prune} acc={model_acc:.4f}"
                )
                acc_dict[float(m)][strategy] = float(model_acc)

        with open(fpath, "wb") as f:
            pickle.dump(acc_dict, f)


# TODO: modularize this function; it is *way* too long right now
# def prune_network(layer_names: str, results_dir: str, device_id: int) -> None:
#    """
#    prune a network with 3 different strategies to test
#    the change in functional performance with each
#    """
#    mreg = np.arange(0, 1, 0.1)
#    thresh = 1e-7
#    pruning_strategies = ["random", "mean_abs", "unbalanced_reg"]
#
#    # loop over all layers for pruning
#    for layer_to_prune in layer_names:
#        fpath = f"{args.results_dir}/{layer_to_prune}.pkl"
#        print(f"pruning layer: {layer_to_prune}")
#
#        if os.path.exists(fpath):
#            continue
#
#        acc_dict = {
#            m: {
#                "mean_abs": None,
#                "unbalanced_reg": None,
#                "random": None,
#                "mean_abs_idx": None,
#                "unbalanced_sm_idx": None,
#                "unbalanced_transform": None,
#            }
#            for m in mreg
#        }
#
#        # get activations for both models
#        activations_model1 = get_activations(
#            model=model1,
#            dloader=dloader,
#            layer_name=layer_to_prune,
#            device_id=device_id,
#        )
#        activations_model2 = get_activations(
#            model=model2,
#            dloader=dloader,
#            layer_name=layer_to_prune,
#            device_id=device_id,
#        )
#
#        # now, loop over all mreg values
#        for m in tqdm(mreg):
#            # compute alignment using partial matching
#            unbalanced_sm = UnbalancedSoftMatch(mass_reg=m)
#            mean_unbalanced_sm = np.mean(
#                np.array(
#                    unbalanced_sm.fit_kfold_score(
#                        activations_model1.cpu(), activations_model2.cpu()
#                    )
#                )
#            )
#            transport_plan = unbalanced_sm.transform
#            neuron_idx = torch.where(transport_plan.sum(axis=1) < thresh)[0]
#            num_units_to_prune = len(neuron_idx)
#            # prune the network weights using each of the pruning techniques
#            for strategy in pruning_strategies:
#                model_copy = copy.deepcopy(model1).to(device_id)
#                submodule = dict(model_copy.named_modules()).get(layer_to_prune)
#
#                if strategy == "mean_abs":
#                    print(f"removing {num_units_to_prune} units")
#                    # compute mean absolute activations over all stimuli
#                    # the goal here is to see which neuron is least activated
#                    # across *all* stimuli. we conjecture this to be functionally
#                    # "irrelevant" or redundant
#                    mean_abs = torch.mean(torch.abs(activations_model1), dim=0)
#                    bottom_channel_idx = torch.argsort(mean_abs)[:num_units_to_prune]
#                    print(f"bottom idx: {bottom_channel_idx}")
#                    acc_dict[m]["mean_abs_idx"] = bottom_channel_idx
#                    # zero-out the least activated kernels
#                    with torch.no_grad():
#                        submodule.weight[bottom_channel_idx, ...] = 0
#
#                elif strategy == "unbalanced_reg":
#                    print(f"removing {num_units_to_prune} units")
#                    acc_dict[m]["unbalanced_sm_idx"] = neuron_idx
#                    acc_dict[m]["unbalanced_transform"] = transport_plan
#                    with torch.no_grad():
#                        submodule.weight[neuron_idx, ...] = 0
#
#                elif strategy == "random":
#                    # print(f"removing {num_units_to_prune} units")
#                    # choose random channels (same as number of neuron indices)
#                    rand_indices = torch.randperm(submodule.weight.shape[0])[
#                        :num_units_to_prune
#                    ]
#                    print(f"rand indices shape: {rand_indices.shape}")
#                    with torch.no_grad():
#                        submodule.weight[rand_indices, ...] = 0
#
#                else:
#                    print(f"invalid pruning strategy: {strategy}")
#
#                # compute model accuracy after each pruning strategy
#                model_acc = compute_accuracy(
#                    model=model_copy, dloader=dloader, device_id=device_id
#                )
#                print(f"strategy: {strategy}\t model acc: {model_acc}")
#                # populate results in dictionary
#                acc_dict[m][strategy] = model_acc
#
#        # save results to disk
#        with open(fpath, "wb") as f:
#            pickle.dump(acc_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", type=str, default=None, help="base path to model weights"
    )
    parser.add_argument(
        "--model_type", type=str, default="resnet18", help="cnn model type for analyses"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../pruning_experiment",
        help="path to results directory",
    )
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # load model weights
    model1_ckpt = os.path.join(args.base_path, f"seed_1/best.pth")
    model2_ckpt = os.path.join(args.base_path, f"seed_2/best.pth")

    # init model instance
    model1 = (
        load_weights(
            ckpt_path=model1_ckpt, model_type=args.model_type, num_classes=1000
        )
        .to(f"cuda:{args.device_id}")
        .eval()
    )
    model2 = (
        load_weights(
            ckpt_path=model2_ckpt, model_type=args.model_type, num_classes=1000
        )
        .to(f"cuda:{args.device_id}")
        .eval()
    )

    # get dataloader instance
    dloader = imagenet_dataloader(dirpath="/mnt/cogsci/KhoslaLab/ILSVRC2012/valid")

    # first, we compute the best test accuracy of both models
    # acc_model1 = compute_accuracy(
    #    model=model1, dloader=dloader, device_id=args.device_id
    # )
    # acc_model2 = compute_accuracy(
    #    model=model2, dloader=dloader, device_id=args.device_id
    # )
    # print(f"model1: {acc_model1}\t model2: {acc_model2}")

    # get all layer names
    layer_names_all = get_layer_names(model=model1)
    layer_names = [l for l in layer_names_all if "conv" in l]

    # prune the network with 3 pruning strategies
    prune_network(
        layer_names=layer_names,
        results_dir=args.results_dir,
        device_id=args.device_id,
        model1=model1,
        model2=model2,
        dloader=dloader,
        args=args,
        use_hook_ablation=True,
    )
    # prune_network(
    #    layer_names=layer_names, results_dir=args.results_dir, device_id=args.device_id
    # )
