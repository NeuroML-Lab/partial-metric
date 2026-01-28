"""
do all nsd-related analyses here
"""

import torch
from tqdm import tqdm

from src.utils.utils import brute_force_matching
from src.utils.nsd_io import load_voxel_responses, gather_responses
from src.analysis.nsd.plots import plot_dropped_nc_unbalanced, overlay_all_baselines
from src.analysis.nsd.noise_ceiling import (
    align_voxel_responses_unbalanced,
    align_voxel_responses_sm_unbalanced,
    compute_noise_ceilings,
    compute_noise_ceilings_sm_unbalanced,
    compute_noise_ceilings_brute_force,
)

brain_regions = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]


def baseline_analysis(base_path: str, subject1: int, subject2: int):
    """
    helper function for conducting all baseline analyses
    """
    for i, region in tqdm(enumerate(brain_regions), total=len(brain_regions)):
        print(f"aligning responses from: {region}")
        responses = load_voxel_responses(base_path=base_path, brain_region=region)
        r1, r2 = gather_responses(data=responses, subject1=subject1, subject2=subject2)

        # partial matching
        print(f"starting partial matching between subjects {subject1} and {subject2}")
        mreg_dict_partial = align_voxel_responses(responses1=r1, responses2=r2)
        mreg_dict_partial = compute_noise_ceilings(
            mreg_dict=mreg_dict_partial,
            base_path=base_path,
            brain_region=region,
            subj1=subject1,
            subj2=subject2,
        )

        # soft-matching
        print(f"starting soft-matching between subject {subject1} and {subject2}")

        # brute force matching
        print(
            f"starting brute force matching between subjects {subject1} and {subject2}"
        )
        # first, get the rank ordering from subject1 -> subject2
        delta_dict_forward = brute_force_matching(torch.tensor(r1), torch.tensor(r2))
        delta_dict_backward = brute_force_matching(torch.tensor(r2), torch.tensor(r1))

        mreg_dict_brute_force = align_voxel_responses_both(responses1=r1, responses2=r2)
        mreg_dict_brute_force = compute_noise_ceilings_brute_force(
            mreg_dict=mreg_dict_brute_force,
            delta_dicts=(delta_dict_forward, delta_dict_backward),
            transform_thresh=1e-8,
            base_path=base_path,
            brain_region=region,
            responses1=r1,
            responses2=r2,
            subj1=subj1,
            subj2=subj2,
        )

    return None


if __name__ == "__main__":
    base_path = "/mnt/cogsci/NSD_preprocessed_datasets_shreya/"
    baseline_analysis(base_path=base_path, subject1=1, subject2=5)
