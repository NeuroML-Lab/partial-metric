"""
here, we define a bunch of helper functions to process and parse
NSD for our brain region mapping experiments
"""

import os
import pickle
from typing import Tuple

import torch
import numpy as np


def load_voxel_responses(base_path: str, brain_region: str) -> dict:
    """
    load voxel responses from a pickle file. every pickle file
    follows the format '{base_path}{brain_region}_data/{brain_region}_data_1257'
    note that we only use data from subjects {1, 2, 5, 7} from the original
    dataset
    """
    with open(
        f"{base_path}{brain_region}_data/{brain_region}_data_1257.pickle", "rb"
    ) as f:
        data = pickle.load(f)

    return data


def gather_responses(
    data: dict, subject1: int, subject2: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    gather common stimulus responses across a subject pair
    """
    subject1_responses, subject2_responses = [], []

    for stimulus_id in data.keys():
        all_subject_responses = data[stimulus_id]
        if (
            all_subject_responses[subj1] is not None
            and all_subject_responses[subj2] is not None
        ):
            subject1_responses.append(all_subject_responses[subj1])
            subject2_responses.append(all_subject_responses[subj2])

    return np.array(subject1_responses), np.array(subject2_responses)
