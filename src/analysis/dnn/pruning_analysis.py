import os
import copy
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch

from src.analysis.dnn.utils import (
    compute_accuracy,
    imagenet_dataloader,
    load_weights,
    get_layer_names,
    get_activations,
)
from src.metrics.alignment.partial_wasserstein import UnbalancedSoftMatch


# TODO: modularize this function; it is *way* too long right now
def prune_network(layer_names: str, results_dir: str, device_id: int) -> None:
    """
    prune a network with 3 different strategies to test
    the change in functional performance with each
    """
    mreg = np.arange(0, 1, 0.1)
    thresh = 1e-7
    pruning_strategies = ["random", "mean_abs", "unbalanced_reg"]

    # loop over all layers for pruning
    for layer_to_prune in layer_names:
        fpath = f"{args.results_dir}/{layer_to_prune}.pkl"
        print(f"pruning layer: {layer_to_prune}")

        if os.path.exists(fpath):
            continue

        acc_dict = {
            m: {
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
            # compute alignment using partial matching
            unbalanced_sm = UnbalancedSoftMatch(mass_reg=m)
            mean_unbalanced_sm = np.mean(
                np.array(
                    unbalanced_sm.fit_kfold_score(
                        activations_model1.cpu(), activations_model2.cpu()
                    )
                )
            )
            transport_plan = unbalanced_sm.transform
            neuron_idx = torch.where(transport_plan.sum(axis=1) < thresh)[0]
            num_units_to_prune = len(neuron_idx)
            # prune the network weights using each of the pruning techniques
            for strategy in pruning_strategies:
                model_copy = copy.deepcopy(model1).to(device_id)
                submodule = dict(model_copy.named_modules()).get(layer_to_prune)

                if strategy == "mean_abs":
                    print(f"removing {num_units_to_prune} units")
                    # compute mean absolute activations over all stimuli
                    # the goal here is to see which neuron is least activated
                    # across *all* stimuli. we conjecture this to be functionally
                    # "irrelevant" or redundant
                    mean_abs = torch.mean(torch.abs(activations_model1), dim=0)
                    bottom_channel_idx = torch.argsort(mean_abs)[:num_units_to_prune]
                    print(f"bottom idx: {bottom_channel_idx}")
                    acc_dict[m]["mean_abs_idx"] = bottom_channel_idx
                    # zero-out the least activated kernels
                    with torch.no_grad():
                        submodule.weight[bottom_channel_idx, ...] = 0

                elif strategy == "unbalanced_reg":
                    print(f"removing {num_units_to_prune} units")
                    acc_dict[m]["unbalanced_sm_idx"] = neuron_idx
                    acc_dict[m]["unbalanced_transform"] = transport_plan
                    with torch.no_grad():
                        submodule.weight[neuron_idx, ...] = 0

                elif strategy == "random":
                    # print(f"removing {num_units_to_prune} units")
                    # choose random channels (same as number of neuron indices)
                    rand_indices = torch.randperm(submodule.weight.shape[0])[
                        :num_units_to_prune
                    ]
                    print(f"rand indices shape: {rand_indices.shape}")
                    with torch.no_grad():
                        submodule.weight[rand_indices, ...] = 0

                else:
                    print(f"invalid pruning strategy: {strategy}")

                # compute model accuracy after each pruning strategy
                model_acc = compute_accuracy(
                    model=model_copy, dloader=dloader, device_id=device_id
                )
                print(f"strategy: {strategy}\t model acc: {model_acc}")
                # populate results in dictionary
                acc_dict[m][strategy] = model_acc

        # save results to disk
        with open(fpath, "wb") as f:
            pickle.dump(acc_dict, f)


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
    # acc_model1 = compute_accuracy(model=model1, dloader=dloader)
    # acc_model2 = compute_accuracy(model=model2, dloader=dloader)
    # print(f"model1: {acc_model1}\t model2: {acc_model2}")

    # get all layer names
    layer_names_all = get_layer_names(model=model1)
    layer_names = [l for l in layer_names_all if "conv" in l]

    # prune the network with 3 pruning strategies
    prune_network(
        layer_names=layer_names, results_dir=args.results_dir, device_id=args.device_id
    )
