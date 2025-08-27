import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from typing import Tuple
from itertools import combinations

from src.metrics.alignment.soft_match import SoftMatch
from src.metrics.alignment.partial_wasserstein import UnbalancedSoftMatch

from src.utils.utils import precision
from src.utils.nsd_io import load_voxel_responses, gather_responses
from src.utils.regularization_search import compute_residuals_and_transport


def stack_responses(
    base_path: str, subject1: int, subject2: int, brain_region1: str, brain_region2: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    stack subject responses from 2 brain regions.
    args:
        subject1: ID of subject1
        subject2: ID of subject2
        brain_region1: first area of visual cortex used
        brain_region2: second area of visual cortex used
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

    # stack subject responses along response dimension
    subject1_stacked = np.hstack((subject1_region1, subject1_region2))
    subject2_stacked = np.hstack((subject2_region1, subject2_region2))

    # get extents of regions (basically response shape limits)
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


def optimal_regularization(responses1: np.ndarray, responses2: np.ndarray) -> float:
    """
    compute the optimal mass regularization for a pair of subject responses
    """
    transport_distance, residuals, mreg = compute_residuals_and_transport(
        matrix1=subject1_stacked,
        matrix2=subject2_stacked,
        num_points=max(subject1_stacked.shape[1], subject2_stacked.shape[1]),
    )
    # estimate the point of inflection
    ddy = np.diff(np.diff(transport_distance))
    inflection_idx = np.argmax(np.abs(ddy)) + 1
    inflection_mreg = mreg[inflection_idx]
    inflection_cost = transport_distance[inflection_idx]
    inflection_residual = residuals[inflection_idx]

    return inflection_mreg


def fit_couplings(responses1: np.ndarray, responses2: np.ndarray, optimal_mreg: float):
    """
    here, we fit the optimal couplings computed via the full and partial
    wasserstein methods. we use these to compute a "precision" score of how
    good/bad regularization serves to match voxels.
    """
    softmatch = SoftMatch()
    unbalanced_soft_match = UnbalancedSoftMatch(mass_reg=optimal_mreg)

    # fit soft-matching
    sm_score = np.mean(np.array(softmatch.fit_kfold_score(responses1, responses2)))
    # fit unbalanced soft-matching
    unsm_score = np.mean(
        np.array(unbalanced_soft_match.fit_kfold_score(responses1, responses2))
    )

    return sm_score, unsm_score, softmatch.transform, unbalanced_soft_match.transform


def compute_coupling_precision(
    transform: torch.Tensor,
    region1_row_max: int,
    region1_col_max: int,
    region2_rows: int,
    region2_cols: int,
) -> float:
    # get region1 block (true positives)
    region1_block = transform[:region1_row_max, :region1_col_max]
    region1_nonzero = np.count_nonzero(region1_block) / (
        region1_row_max + region1_col_max
    )

    # get region2 block (true positives)
    region2_block = transform[-region2_rows:, -region2_cols:]
    region2_nonzero = np.count_nonzero(region2_block) / (region2_rows + region2_cols)

    # top-right block (false positives)
    top_right = transform[:region1_row_max, region1_col_max:]
    top_right_nonzero = np.count_nonzero(top_right) / (
        top_right.shape[0] + top_right.shape[1]
    )

    # bottom-left block (false positives)
    bottom_left = transform[region1_row_max:, :region1_col_max]
    bottom_left_nonzero = np.count_nonzero(bottom_left) / (
        bottom_left.shape[0] + bottom_left.shape[1]
    )

    transport_precision = precision(
        true_positive=(region1_nonzero + region1_nonzero),
        false_positive=(top_right_nonzero + bottom_left_nonzero),
    )

    return transport_precision


if __name__ == "__main__":
    base_path = "/mnt/cogsci/NSD_preprocessed_datasets_shreya/"
    subject1, subject2 = 1, 2
    brain_regions = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]

    # construct all possible brain region combination pairs
    region_pairs = list(combinations(brain_regions, 2))

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
        ) = stack_responses(
            base_path=base_path,
            subject1=subject1,
            subject2=subject2,
            brain_region1=brain_region1,
            brain_region2=brain_region2,
        )

        # get optimal regularization parameter
        inflection_mreg = optimal_regularization(
            responses1=subject1_stacked, responses2=subject2_stacked
        )
        print(f"inflection mreg: {inflection_mreg}")

        # fit optimal transport couplings
        sm_score, unsm_score, sm_transform, unsm_transform = fit_couplings(
            responses1=subject1_stacked,
            responses2=subject2_stacked,
            optimal_mreg=inflection_mreg,
        )

        # compute coupling precisions
        sm_precision = compute_coupling_precision(
            transform=sm_transform,
            region1_row_max=region1_row_max,
            region1_col_max=region1_col_max,
            region2_rows=region2_rows,
            region2_cols=region2_cols,
        )
        unsm_precision = compute_coupling_precision(
            transform=unsm_transform,
            region1_row_max=region1_row_max,
            region1_col_max=region1_col_max,
            region2_rows=region2_rows,
            region2_cols=region2_cols,
        )

        print(
            f"{brain_region1} + {brain_region2}:\t sm score: {sm_score:.3f}   sm prec: {sm_precision:.3f}   unsm score: {unsm_score:.3f}   unsm prec: {unsm_precision:.3f}"
        )
