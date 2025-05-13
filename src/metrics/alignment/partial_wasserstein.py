import ot
import warnings
import numpy as np
from typing import Optional, List, Union

from sklearn.model_selection import KFold
from src.metrics.base import Metric

import torch
from torchmetrics.functional import pairwise_cosine_similarity


class UnbalancedSoftMatch(Metric):
    """
    compute a partial wasserstein transport plan to align a
    distribution pair (eg: a pair of neural activations)
    """

    def __init__(self, mass_reg: Optional[float] = 1e-1, k_folds: Optional[int] = 5):
        super().__init__(k_folds=k_folds)
        self.mass_reg = mass_reg

    @staticmethod
    def _clean_matrices(x: torch.Tensor) -> torch.Tensor:
        """
        remove all 0-columns from x.
        """
        rem = torch.all(x == 0, dim=0).nonzero(as_tuple=True)[0]
        keep_indices = torch.tensor(
            [i for i in range(x.shape[1]) if i not in rem.cpu()], device=x.device
        )

        return torch.index_select(x, 1, keep_indices)

    def fit(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> "UnbalancedSoftMatch":
        x = torch.tensor(x) if isinstance(x, np.ndarray) else x
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y

        x = self._clean_matrices(x)
        y = self._clean_matrices(y)

        # mean center responses
        self.mx_ = torch.nanmean(x.T, dim=1)[:, None]
        self.my_ = torch.nanmean(y.T, dim=1)[:, None]
        x_centered = x.T - self.mx_
        y_centered = y.T - self.my_

        # compute cost matrix
        self.cost = 1 - pairwise_cosine_similarity(x_centered, y_centered)

        # initialize uniform marginals
        a = torch.ones(x.shape[1], device=x.device) / x.shape[1]
        b = torch.ones(y.shape[1], device=x.device) / y.shape[1]

        self.transform = ot.partial.partial_wasserstein(
            a.cpu(), b.cpu(), self.cost.cpu(), m=float(self.mass_reg)
        ).to(x_centered.device)
        return self

    def score(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> float:
        """
        compute distance as a similarity score
        """
        use_kfold = kwargs.get("use_kfold", False)

        x = torch.tensor(x) if isinstance(x, np.ndarray) else x
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y

        # remove all 0-columns
        x = self._clean_matrices(x)
        y = self._clean_matrices(y)

        if not use_kfold:
            x = x.T - self.mx_
            y = y.T - self.my_
            similarity_score = 1 - torch.nansum(self.transform * self.cost).item()

        else:
            mx = torch.nanmean(x.T, dim=1)[:, None]
            my = torch.nanmean(y.T, dim=1)[:, None]
            x_centered = x.T - mx
            y_centered = y.T - my
            # compute cost of transport in test split
            cost = 1 - pairwise_cosine_similarity(x_centered, y_centered)
            similarity_score = 1 - torch.nansum(self.transform * cost).item()

        return similarity_score


if __name__ == "__main__":
    x = torch.randn(100, 20).cuda()
    y = torch.randn(100, 20).cuda()

    metric = UnbalancedSoftMatch(mass_reg=0.5)
    metric2 = UnbalancedSoftMatch_orig(mass_reg=0.5)

    print(f"unbalanced metric: {metric.fit_kfold_score(x=x, y=y)}")
    print(f"unbalanced metric: {metric2.fit_kfold_score(x=x.cpu(), y=y.cpu())}")
