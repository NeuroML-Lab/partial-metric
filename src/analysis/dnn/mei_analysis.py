import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

from src.metrics.alignment.partial_wasserstein import UnbalancedSoftMatch


def align_activations(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    step_size: Optional[float] = 0.1,
) -> dict:
    """
    align activations over all mreg values between [0, 1].

    params:
        activations1 (torch.Tensor): activations of feature maps in the first network
        activations2 (torch.Tensor): activations of feature maps in the second network
        step_size (float): step size for deciding the spacing of mreg values

    returns:
        mreg_dict (dict): dictionary of alignment scores and transport plans with their
                          corresponding mreg values
    """
    mreg_values = np.arange(0, 1.1, step_size)
    mreg_values = np.where(
        mreg_values == 1.0, 0.99, mreg_values
    )  # replace the last value with 1

    # init an empty dictionary
    mreg_dict = {m: {"score": None, "transform": None} for m in mreg_values}

    for m in tqdm(mreg_dict.keys()):
        metric = UnbalancedSoftMatch(mass_reg=m)
        score = np.mean(
            np.array(
                metric.fit_kfold_score(
                    torch.tensor(activations1), torch.tensor(activations2)
                )
            )
        )
        mreg_dict[m]["score"] = score
        mreg_dict[m]["transform"] = metric.transform

    return mreg_dict


def best_mutual_matches(
    transport_plan: torch.Tensor, threshold: float = 1e-3, return_weights: bool = False
) -> List[Tuple[int, int, float]]:
    """
    get mutual-best matches from the transport plan, ordered by transport mass (descending)

    params:
        transport_plan (torch.Tensor): 2D tensor of shape (n_src, n_tgt) with nonnegative transport masses
        threshold (float): minimum weight to consider a match
        return_weights (bool): if True return list of (src_idx, tgt_idx, weight);
                               if False return list of (src_idx, tgt_idx)

    returns:
        matches_sorted (list): sorted by weight (largest first)
    """
    n_src, n_target = transport_plan.shape

    src_best = torch.argmax(transport_plan, dim=1)  # best target per source
    target_best = torch.argmax(transport_plan, dim=0)  # best source per target

    matches = []
    for i in range(n_src):
        j = int(src_best[i].item())  # best (source, target) match
        transport_weight = float(
            transport_plan[i, j].item()
        )  # corresponding transport weight

        if transport_weight > threshold and int(target_best[j].item()) == i:
            matches.append((int(i), int(j), transport_weight))

    # sort matches in descending order of transport weight
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

    if return_weights:
        return sorted_matches
    else:
        return [(i, j) for (i, j, _) in sorted_matches]


def plot_mei_matches(
    matches: List[Tuple[int, int]],
    layer_name: str,
    model_type: str,
    base_path: str,
    figsize_per_row: Tuple[float, float] = (6, 2.5),
    save_plot: bool = False,
    save_path: Optional = None,
):
    """
    plot matched MEIs, with one pair per row
    """
    num_matches = len(matches)

    fig_w = figsize_per_row[0]
    fig_h = figsize_per_row[1] * num_matches

    fig, axes = plt.subplots(nrows=num_matches, ncols=2, figsize=(fig_w, fig_h))

    # loop over all matches
    for row, (i, j) in enumerate(matches):
        mei1 = plt.imread(
            f"{base_path}/{model_type}_s1/{layer_name}/channel_{i}_center.png"
        )
        mei2 = plt.imread(
            f"{base_path}/{model_type}_s2/{layer_name}/channel_{j}_center.png"
        )

        # left image
        ax_left = axes[row, 0]
        ax_left.imshow(mei1)
        ax_left.set_title(rf"Channel ${i}$", fontsize=10)
        ax_left.axis("off")

        # right image
        ax_right = axes[row, 1]
        ax_right.imshow(mei2)
        ax_right.set_title(rf"Channel ${j}$", fontsize=10)
        ax_right.axis("off")

    plt.tight_layout()

    if save_plot:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(
            f"{save_path}/{model_type}_{layer_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
    else:
        plt.show()
