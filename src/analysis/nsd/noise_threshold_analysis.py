import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
from itertools import combinations

from src.metrics.alignment.soft_match import SoftMatch

from src.utils.utils import precision
from src.utils.nsd_io import load_voxel_responses, gather_responses
from src.analysis.nsd.l_curve_analysis import compute_coupling_precision


def threshold_and_stack_responses(
    base_path: str,
    subject1: int,
    subject2: int,
    brain_region1: str,
    brain_region2: str,
    noise_ceiling_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    stack responses from 2 brain regions and threshold
    based on noise ceiling of recorded voxels
    """
    # load voxel responses from 2 different brain regions
    responses_region1 = load_voxel_responses(
        base_path=base_path, brain_region=brain_region1
    )
    responses_region2 = load_voxel_responses(
        base_path=base_path, brain_region=brain_region2
    )

    # now, load responses for specific subject IDs
    subject1_region1, subject2_region1 = gather_responses(
        data=responses_region1, subject1=subject1, subject2=subject2
    )
    subject1_region2, subject2_region2 = gather_responses(
        data=responses_region2, subject1=subject1, subject2=subject2
    )

    # load noise ceiling files
    nc_region1_subj1 = np.load(f"{base_path}{brain_region1}_data/nc_0{subject1}.npy")
    nc_region2_subj1 = np.load(f"{base_path}{brain_region2}_data/nc_0{subject1}.npy")
    nc_region1_subj2 = np.load(f"{base_path}{brain_region1}_data/nc_0{subject2}.npy")
    nc_region2_subj2 = np.load(f"{base_path}{brain_region2}_data/nc_0{subject2}.npy")

    # find indices of low noise ceiling
    region1_subj1_idx = np.where(nc_region1_subj1 < noise_ceiling_thresh)[0]
    region2_subj1_idx = np.where(nc_region2_subj1 < noise_ceiling_thresh)[0]
    region1_subj2_idx = np.where(nc_region1_subj2 < noise_ceiling_thresh)[0]
    region2_subj2_idx = np.where(nc_region2_subj2 < noise_ceiling_thresh)[0]

    fraction_kept = 1 - (
        len(region1_subj1_idx)
        + len(region1_subj2_idx)
        + len(region2_subj1_idx)
        + len(region2_subj2_idx)
    ) / (
        subject1_region1.shape[-1]
        + subject1_region2.shape[-1]
        + subject2_region1.shape[-1]
        + subject2_region2.shape[-1]
    )

    print(f"fraction of kept voxels: {fraction_kept*100:.2f}")

    # delete voxel responses which fall below the threshold
    subject1_region1, subject1_region2 = np.delete(
        subject1_region1, region1_subj1_idx, axis=1
    ), np.delete(subject1_region2, region2_subj1_idx, axis=1)

    subject2_region1, subject2_region2 = np.delete(
        subject2_region1, region1_subj2_idx, axis=1
    ), np.delete(subject2_region2, region2_subj2_idx, axis=1)

    # stack responses
    subject1_stacked = np.hstack((subject1_region1, subject1_region2))
    subject2_stacked = np.hstack((subject2_region1, subject2_region2))

    # get region limits for each subject
    region1_row_max, region1_col_max = (
        subject1_region1.shape[-1],
        subject2_region1.shape[-1],
    )
    region2_rows, region2_cols = subject1_region2.shape[-1], subject2_region2.shape[-1]

    return (
        subject1_stacked,
        subject2_stacked,
        region1_row_max,
        region1_col_max,
        region2_rows,
        region2_cols,
    )


def fit_soft_match_coupling(responses1: np.ndarray, responses2: np.ndarray):
    softmatch = SoftMatch()
    # fit soft-matching
    sm_score = np.mean(np.array(softmatch.fit_kfold_score(responses1, responses2)))
    return sm_score, softmatch.transform


if __name__ == "__main__":
    base_path = "/mnt/cogsci/NSD_preprocessed_datasets_shreya/"
    subject1, subject2 = 1, 2
    brain_regions = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]
    noise_ceiling_threshold = 0.3

    # construct all possible brain region combination pairs
    region_pairs = list(combinations(brain_regions, 2))
    soft_match = SoftMatch()

    print(f"using noise ceiling threshold = {noise_ceiling_threshold}")
    for brain_region1, brain_region2 in tqdm(region_pairs, total=len(region_pairs)):
        print(f"using responses from {brain_region1} and {brain_region2}")
        # gather subject responses
        (
            subject1_stacked,
            subject2_stacked,
            region1_row_max,
            region1_col_max,
            region2_rows,
            region2_cols,
        ) = threshold_and_stack_responses(
            base_path=base_path,
            subject1=subject1,
            subject2=subject2,
            brain_region1=brain_region1,
            brain_region2=brain_region2,
            noise_ceiling_thresh=noise_ceiling_threshold,
        )

        # fit optimal soft-match coupling
        sm_score, sm_transform = fit_soft_match_coupling(
            responses1=subject1_stacked, responses2=subject2_stacked
        )

        # compute coupling precision
        sm_precision = compute_coupling_precision(
            transform=sm_transform,
            region1_row_max=region1_row_max,
            region1_col_max=region1_col_max,
            region2_rows=region2_rows,
            region2_cols=region2_cols,
        )

        print(
            f"{brain_region1} + {brain_region2}:\t sm score: {sm_score:.3f}   sm prec: {sm_precision:.3f}"
        )
