"""
helper functions for NSD noise ceiling experiments
"""

import torch
import numpy as np
from tqdm import tqdm

from src.utils.utils import pairwise_correlation
from src.metrics.alignment.soft_match import SoftMatch
from src.metrics.alignment.partial_wasserstein import UnbalancedSoftMatch


def align_voxel_responses_unbalanced(
    responses1: np.ndarray, responses2: np.ndarray
) -> dict:
    """
    align voxel responses from a subject pair using partial matching
    across a range of different regularization parameters
    """
    mreg_values = np.arrange(0, 1.1, 0.1)
    mreg_values = np.where(mreg_values == 1.0, 0.99, mreg_values)
    mreg_dict = {
        m: {
            "score": None,  # alignment score
            "transform": None,  # optimal transport plan
            "mean_dropped_nc_subj1": None,  # mean noise ceiling of dropped voxels
            "mean_dropped_nc_subj2": None,
            "mean_nc_subj1": None,  # mean noise ceiling of retained voxels
            "mean_nc_subj2": None,
        }
        for m in mreg_values
    }

    for m in tqdm(mreg_dict.keys()):
        metric = UnbalancedSoftMatch(mass_reg=m)
        score = np.mean(
            np.array(
                metric.fit_score(torch.tensor(responses1), torch.tensor(responses2))
            )
        )
        mreg_dict[m]["score"] = score
        mreg_dict[m]["transform"] = metric.transform

    return mreg_dict


def align_voxel_responses_sm_unbalanced(
    responses1: np.ndarray, responses2: np.ndarray
) -> dict:
    """
    align voxel responses of a subject pair using soft-matching and partial matching
    """
    mreg_values = np.arange(0, 1.1, 0.1)
    mreg_values = np.where(mreg_values == 1.0, 0.99, mreg_values)
    mreg_dict = {
        m: {
            "unbalanced_score": None,
            "sm_score": None,
            "unbalanced_transform": None,
            "sm_transform": None,
            "mean_dropped_nc_subj1": None,
            "mean_dropped_nc_subj2": None,
            "mean_nc_subj1": None,
            "mean_nc_subj2": None,
        }
        for m in mreg_values
    }

    soft_match = SoftMatch()
    sm_score = np.mean(
        np.array(
            soft_match.fit_score(torch.tensor(responses1), torch.tensor(responses2))
        )
    )

    for m in tqdm(mreg_dict.keys()):
        unbalanced_sm = UnbalancedSoftMatch(mass_reg=m)
        score = np.mean(
            np.array(
                unbalanced_sm.fit_score(
                    torch.tensor(responses1), torch.tensor(responses2)
                )
            )
        )
        mreg_dict[m]["unbalanced_score"] = score
        mreg_dict[m]["sm_score"] = sm_score
        mreg_dict[m]["unbalanced_transform"] = unbalanced_sm.transform
        mreg_dict[m]["sm_transform"] = soft_match.transform

    return mreg_dict


def compute_noise_ceilings(
    mreg_dict: dict,
    base_path: str,
    brain_region: str,
    subj1: int,
    subj2: int,
    transform_thresh: float = 1e-8,
) -> dict:
    nc_subject1 = np.load(f"{base_path}{brain_region}_data/nc_0{subj1}.npy")
    nc_subject2 = np.load(f"{base_path}{brain_region}_data/nc_0{subj2}.npy")

    # find voxels below noise_thresh
    below_nc_subj1 = nc_subject1[nc_subject1 < noise_thresh]
    below_nc_subj2 = nc_subject2[nc_subject2 < noise_thresh]

    for e in nc_subject1:
        if e < 0:
            print(e)

    # compute mean noise ceiling of voxels below the threshold
    mean_subj1, mean_subj2 = below_nc_subj1.mean(), below_nc_subj2.mean()

    for mreg in mreg_dict.keys():
        dropped_units_subj1 = torch.where(
            mreg_dict[mreg]["transform"].sum(axis=1) < transform_thresh
        )[0].numpy()
        dropped_units_subj2 = torch.where(
            mreg_dict[mreg]["transform"].sum(axis=0) < transform_thresh
        )[0].numpy()

        if len(dropped_units_subj1) > 0:
            mean_nc_subj1 = nc_subject1[dropped_units_subj1].mean()
        else:
            mean_nc_subj1 = 0

        if len(dropped_units_subj2) > 0:
            mean_nc_subj2 = nc_subject2[dropped_units_subj2].mean()
        else:
            mean_nc_subj2 = 0

        # mean noise ceiling of dropped voxels from the transport plan
        mreg_dict[mreg]["mean_dropped_nc_subj1"] = mean_nc_subj1
        mreg_dict[mreg]["mean_dropped_nc_subj2"] = mean_nc_subj2

        # mean noise ceiling of voxels below noise_thresh
        mreg_dict[mreg]["mean_nc_subj1"] = mean_subj1
        mreg_dict[mreg]["mean_nc_subj2"] = mean_subj2

    return mreg_dict


def compute_noise_ceilings_sm_unbalanced(
    mreg_dict: dict,
    noise_thresh: float,
    transform_thresh: float,
    base_path: str,
    brain_region: str,
    responses1: np.ndarray,
    responses2: np.ndarray,
    subj1: int,
    subj2: int,
):
    """
    compute noise ceilings of retained voxels using soft-matching and partial
    wasserstein distance metric.
    """
    nc1 = np.load(f"{base_path}{brain_region}_data/nc_0{subj1}.npy")
    nc2 = np.load(f"{base_path}{brain_region}_data/nc_0{subj2}.npy")

    # mean noise-ceiling of voxels *below* noise_thresh
    low1 = nc1[nc1 < noise_thresh]
    low2 = nc2[nc2 < noise_thresh]
    mean_low1 = low1.mean() if low1.size else np.nan
    mean_low2 = low2.mean() if low2.size else np.nan

    n_vox1, n_vox2 = responses1.shape[-1], responses2.shape[-1]

    softmatch = SoftMatch()

    # loop over all mreg values
    for mreg, data in mreg_dict.items():
        U = data["unbalanced_transform"]  # [n_vox1 x n_vox2]
        SM = data["sm_transform"]  # [n_vox1 x n_vox2]

        # sanity check
        assert U.shape == (n_vox1, n_vox2)
        assert SM.shape == (n_vox1, n_vox2)

        # how many voxels are kept using partial matching?
        kept1 = torch.where(U.sum(dim=1) >= transform_thresh)[0]
        kept2 = torch.where(U.sum(dim=0) >= transform_thresh)[0]
        k1, k2 = kept1.numel(), kept2.numel()

        # transform responses from subj1 -> subj2
        t12 = torch.tensor(responses1, dtype=torch.float32) @ SM  # [n_stim, n_vox2]
        corr12 = pairwise_correlation(
            t12, torch.tensor(responses2, dtype=torch.float32)
        )  # [n_vox2]
        kept_2 = corr12.argsort(descending=True)[:k2]
        dropped_2 = corr12.argsort(descending=False)[: n_vox2 - k2]

        # transform responses from subj2 -> subj1
        t21 = torch.tensor(responses2, dtype=torch.float32) @ SM.T  # [n_stim, n_vox1]
        corr21 = pairwise_correlation(
            t21, torch.tensor(responses1, dtype=torch.float32)
        )  # [n_vox1]
        kept_1 = corr21.argsort(descending=True)[:k1]
        dropped_1 = corr21.argsort(descending=False)[: n_vox1 - k1]

        # compute mean noise ceiling of dropped indices
        mean_drop_nc1 = nc1[list(dropped_1)].mean() if len(list(dropped_1)) > 0 else 0
        mean_drop_nc2 = nc2[list(dropped_2)].mean() if len(list(dropped_2)) > 0 else 0

        # select voxel indices with higher correlation
        sub1 = torch.tensor(r1)[:, kept_1]
        sub2 = torch.tensor(r2)[:, kept_2]

        # recompute soft-match on the remaining voxels
        if sub1.shape[-1] == 0 or sub2.shape[-1] == 0:
            sm_score = 1
        else:
            sm = SoftMatch()
            # sm_score = np.mean(np.array(sm.fit_kfold_score(sub1, sub2)))
            sm_score = np.mean(np.array(sm.fit_score(sub1, sub2)))

        data["mean_nc_subj1"] = mean_low1
        data["mean_nc_subj2"] = mean_low2
        data["mean_dropped_nc_subj1"] = mean_drop_nc1
        data["mean_dropped_nc_subj2"] = mean_drop_nc2
        data["sm_score"] = sm_score

    return mreg_dict


def compute_noise_ceilings_brute_force(
    mreg_dict: dict,
    delta_dicts: Tuple[dict, dict],
    transform_thresh: float,
    base_path: str,
    brain_region: str,
    responses1: np.ndarray,
    responses2: np.ndarray,
    subj1: int,
    subj2: int,
):
    """
    compute the noise ceilings of retained voxels using the brute-force
    neuron ordering approach.
    """
    nc1 = np.load(f"{base_path}{brain_region}_data/nc_0{subj1}.npy")
    nc2 = np.load(f"{base_path}{brain_region}_data/nc_0{subj2}.npy")

    delta_dict_forward, delta_dict_backward = delta_dicts
    forward_index_ordering = list(delta_dict_forward.keys())
    backward_index_ordering = list(delta_dict_backward.keys())

    n_vox1, n_vox2 = responses1.shape[-1], responses2.shape[-1]

    softmatch = SoftMatch()

    # loop over all mreg values
    for mreg, data in mreg_dict.items():
        U = data["unbalanced_transform"]  # [n_vox1 x n_vox2]
        SM = data["sm_transform"]

        # sanity check
        assert U.shape == (n_vox1, n_vox2)

        # how many voxels are kept using partial matching?
        kept1 = torch.where(U.sum(dim=1) >= transform_thresh)[0]
        kept2 = torch.where(U.sum(dim=0) >= transform_thresh)[0]
        k1, k2 = kept1.numel(), kept2.numel()

        # augment responses to refit soft-matching
        print(max(forward_index_ordering), max(backward_index_ordering), SM.shape)

        if k1 == 0 or k2 == 0:
            sm_score = 1
        else:
            _ = softmatch.fit(responses1, responses2)
            augmented_sm_transform = softmatch.transform.detach().clone()
            augmented_sm_transform[forward_index_ordering[::-1][: n_vox1 - k1], :] = 0
            augmented_sm_transform[:, backward_index_ordering[::-1][: n_vox2 - k2]] = 0
            sm_score = 1 - torch.nansum(augmented_sm_transform * softmatch.cost).item()

        print(f"score: {sm_score}")
        # mean noise ceilings of dropped indices
        dropped_1 = forward_index_ordering[::-1][: n_vox1 - k1]
        dropped_2 = backward_index_ordering[::-1][: n_vox2 - k2]
        mean_drop_nc1 = nc1[list(dropped_1)].mean() if len(list(dropped_1)) > 0 else 0
        mean_drop_nc2 = nc2[list(dropped_2)].mean() if len(list(dropped_2)) > 0 else 0

        print(mean_drop_nc1, mean_drop_nc2)

        data["mean_dropped_nc_subj1"] = mean_drop_nc1
        data["mean_dropped_nc_subj2"] = mean_drop_nc2
        data["sm_score"] = sm_score

    return mreg_dict
