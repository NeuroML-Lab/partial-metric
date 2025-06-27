import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

from src.metrics.alignment.partial_wasserstein import UnbalancedSoftMatch


def compute_curvature(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    for a parametrized function f(x, y), compute the curvature using
    finite differences
    """
    eps = 1e-8  # constant for numerical stability
    dx, dy = torch.gradient(x), torch.gradient(y)
    ddx, ddy = torch.gradient(dx), torch.gradient(dy)
    curvature = torch.abs(dx * ddy - ddx * dy) / ((dx**2 + dy**2) ** 1.5 + eps)

    return curvature


def compute_corner(curvatures: torch.Tensor) -> torch.Tensor:
    """
    a heuristic in L-curve methods is to find where the curvature
    is highest. this is where we have an optimal tradeoff between
    fitting and regularization.
    """
    return torch.argmax(curvatures)


def compute_residuals_and_transport(
    matrix1: torch.Tensor, matrix2: torch.Tensor, num_points: int
) -> Tuple[torch.float64, torch.float64, torch.Tensor]:
    """
    compute the residuals (in our case, untransported mass) and their
    associated transport costs (1 - score) or dissimilarity. generally
    speaking, more mass reg parameters would be better since we are closer
    to approximating a continuous curve, giving better curvature approximations.
    """
    transport_costs, residuals = [], []
    mreg_values = torch.linspace(0, 0.999, num_points)

    for m in tqdm(mreg_values):
        # instantiate the metric
        metric = UnbalancedSoftMatch(mass_reg=m)

        scores = metric.fit_kfold_score(x=matrix1, y=matrix2)
        transport_cost = torch.mean(torch.tensor(scores))
        transport_costs.append(transport_cost)

        transport_plan = metric.transform
        # compute the total transported mass
        transported_mass = transport_plan.sum().item()

        # compute the residual (unmatched mass)
        # note that we have unit normalized the sum of inputs
        residual = 1 - transported_mass
        residuals.append(residual)

    return transport_costs, residuals, mreg_values
