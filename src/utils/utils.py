import torch
import numpy as np
from tqdm import tqdm

from src.metrics.alignment.soft_match import SoftMatch


def precision(true_positive: float, false_positive: float) -> float:
    """
    compute a precision score given the number of true and false
    positive measures
    """
    return true_positive / (true_positive + false_positive)


def pairwise_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    compute the pairwise correlation between x and y
    """
    xm = x - torch.mean(x, dim=0, keepdim=True)
    ym = y - torch.mean(y, dim=0, keepdim=True)

    corr_matrix = (xm.T @ ym) / (
        torch.sqrt(torch.sum(xm**2, dim=0, keepdim=True)).T
        * torch.sqrt(torch.sum(ym**2, dim=0, keepdim=True))
    )

    # we care about unit-unit matching
    return torch.diag(corr_matrix)


def brute_force_matching(z1: torch.Tensor, z2: torch.Tensor) -> dict:
    """
    brute force matching of neurons in a representaion pair
    """
    # we want to store the indices of the neurons deleted
    delta_dict = {k: None for k in range(z1.shape[-1])}
    sm = SoftMatch()
    mean_sm = np.mean(np.array(sm.fit_score(z1, z2)))

    # remove each neuron and see how the score is impacted
    # we want to repeat this over N neurons
    for neuron_idx in tqdm(range(z1.shape[-1]), position=0, leave=True):
        if neuron_idx == 0:
            candidate = z1[:, 1:]
        elif neuron_idx == z1.shape[1] - 1:
            candidate = z1[:, :-1]
        else:
            candidate = torch.cat([z1[:, :neuron_idx], z1[:, neuron_idx + 1 :]], dim=1)

        # refit soft-matching to the augmented representation
        score = torch.tensor(sm.fit_score(candidate, z2))
        new_score = score
        delta = (new_score - mean_sm).item()

        delta_dict[neuron_idx] = delta

    # return a (descending) rank-ordered list of units in terms of decrease in correlation
    sorted_delta_dict = {
        k: v
        for k, v in sorted(delta_dict.items(), key=lambda item: item[1], reverse=False)
    }

    return sorted_delta_dict
