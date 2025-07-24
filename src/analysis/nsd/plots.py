import os
import numpy as np
from tqdm import tqdm
from typing import Tuple

import seaborn as sns
import matplotlib.pyplot as plt

# personal plotting pet peeves
plt.rcParams["text.usetex"] = True

from src.experiments.nsd.noise_ceiling import (
    align_voxel_responses_unbalanced,
    compute_noise_ceilings,
)
from src.utils.nsd_io import load_voxel_responses, gather_responses

# define brain regions for use in all plotting functions
# this is per the NSD key-value naming conventions
brain_regions = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]


def plot_dropped_nc_unbalanced(
    subject1: int,
    subject2: int,
    save_path: str = None,
    nrows: int = 2,
    ncols: int = 3,
    figsize: Tuple[float, float] = (9.6, 6.4),
):
    """
    plot the mean noise ceiling of dropped voxels in a subject pair using
    unbalanced matching
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # align responses from all brain regions
    for i, region in tqdm(enumerate(brain_regions), total=len(brain_regions)):
        # load all voxel responses
        responses = load_voxel_responses(base_path=base_path, brain_region=region)
        # gather responses for a subject pair
        r1, r2 = gather_responses(data=responses, subj1=subject1, subj2=subject2)
        # align voxel responses using partial wasserstein
        mreg_dict_partial = align_voxel_responses(responses1=r1, responses2=r2)
        mreg_dict_partial = compute_noise_ceilings(
            mreg_dict=mreg_dict_partial,
            noise_thresh=0.4,
            transform_thresh=1e-8,
            base_path=base_path,
            brain_region=region,
            subj1=subj1,
            subj2=subj2,
        )

        mreg_values = np.array(sorted(mreg_dict_partial.keys()))
        alignment_scores = np.array(
            [mreg_dict_partial[m]["score"] for m in mreg_values]
        )
        # use the computed mean noise ceilings
        mean_nc_subj1 = np.array(
            [mreg_dict_partial[m]["mean_dropped_nc_subj1"] for m in mreg_values]
        )
        mean_nc_subj2 = np.array(
            [mreg_dict_partial[m]["mean_dropped_nc_subj2"] for m in mreg_values]
        )

        ax1 = axes[i]  # select subplot

        # left y-axis: alignment score
        color1 = "tab:blue"
        if i > 2:
            ax1.set_xlabel(r"$m_\mathrm{reg}$")
        if i == 0 or i == 3:
            ax1.set_ylabel("Alignment Score", color=color1, fontsize=10)
        ax1.plot(
            mreg_values,
            alignment_scores,
            marker="o",
            linestyle="-",
            color=color1,
            label="Alignment Score",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.spines["top"].set_visible(False)
        # auc = np.trapezoid(alignment_scores, mreg_values)
        # ax1.text(0.98, 0.95, fr"AUC: ${auc:.2f}$", transform=ax1.transAxes,
        #        ha='right', va='top', color=color1, fontsize=9)

        # right y-axis: mean noise ceiling values
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        if i == 2 or i == 5:
            ax2.set_ylabel(
                "Mean Noise Threshold", rotation=-90, labelpad=15, fontsize=10
            )

        color2 = "tab:red"
        color3 = "tab:green"
        ax2.plot(
            mreg_values,
            mean_nc_subj1,
            marker="s",
            linestyle="--",
            color=color2,
            label=rf"Subject ${subject1}$",
        )
        ax2.plot(
            mreg_values,
            mean_nc_subj2,
            marker="d",
            linestyle="--",
            color=color3,
            label=rf"Subject ${subject2}$",
        )
        ax2.tick_params(axis="y")

        # brain region name
        ax1.set_title(region, fontsize=12, fontweight="bold")

        # combine legends (only in the first subplot)
        if i == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f"{save_path}/dropped_unbalanced_noise_ceiling.png", dpi=300)


def overlay_all_baselines(
    mreg_dict_partial: dict,
    mreg_dict_corr: dict,
    mreg_dict_brute_force: dict,
    nrows: int = 2,
    ncols: int = 3,
    figsize: Tuple[float, float] = (12 / 1.25, 8 / 1.25),
    save_path: str = None,
):
    """
    plot an overlay of correlation scores using 3 approaches:
    - soft-matching (ordered using correlation)
    - brute-force matching
    - partial wasserstein
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, region in tqdm(enumerate(brain_regions), total=len(brain_regions)):
        mreg_values = np.array(sorted(mreg_dict_partial.keys()))
        alignment_partial = np.array(
            [mreg_dict_partial[m]["score"] for m in mreg_values]
        )
        alignment_corr = np.array([mreg_dict_corr[m]["sm_score"] for m in mreg_values])
        alignment_brute_force = np.array(
            [mreg_dict_brute_force[m]["sm_score"] for m in mreg_values]
        )

        ax1 = axes[i]  # select subplot

        # left y-axis: alignment score
        if i > 2:
            axes[i].set_xlabel(r"$m_\mathrm{reg}$")
        if i == 0 or i == 3:
            axes[i].set_ylabel("Alignment Score", fontsize=10)
        axes[i].plot(
            mreg_values,
            alignment_partial,
            marker="o",
            linestyle="-",
            label="Partial Matching",
        )
        axes[i].plot(
            mreg_values,
            alignment_corr,
            marker="s",
            linestyle="-",
            label="Correlation Matching",
        )
        axes[i].plot(
            mreg_values,
            alignment_brute_force,
            marker="^",
            linestyle="-",
            label="Brute Force Matching",
        )

        # brain region name
        axes[i].set_title(region, fontsize=12, fontweight="bold")

        # combine legends (only in the first subplot)
        if i == 0:
            lines1, labels1 = axes[i].get_legend_handles_labels()
            axes[i].legend(lines1, labels1, loc="best", fontsize=8)

    fig.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(f"{save_path}/correlation_overlay_all.png", dpi=300)
