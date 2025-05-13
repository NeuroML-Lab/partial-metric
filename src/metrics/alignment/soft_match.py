import ot
import numpy as np
from typing import Optional, List, Union

from src.metrics.base import Metric

import torch
from torchmetrics.functional import pairwise_cosine_similarity


class SoftMatch(Metric):
    """
    compute an optimal soft-permutation matrix to align a pair
    of neural activations
    """

    def __init__(self, k_folds: Optional[int] = 5):
        super().__init__(k_folds=k_folds)

    @staticmethod
    def _clean_matrices(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
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
    ) -> "SoftMatch":
        """
        compute optimal soft-permutation matrix
        """
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

        # compute optimal transport plan using earth movers distance
        self.transform = ot.emd(a, b, self.cost).to(x.device)

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
        use_kfold = kwargs.get("use_kfold", True)

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

        # assert that we have a birkhoff
        norm_factor_row = 1 / self.transform.shape[0]
        norm_factor_col = 1 / self.transform.shape[1]
        row_sum, col_sum = torch.sum(self.transform, dim=1), torch.sum(
            self.transform, dim=0
        )
        row_check = torch.all(torch.abs(row_sum - norm_factor_row) < 1e-3)
        col_check = torch.all(torch.abs(col_sum - norm_factor_col) < 1e-3)
        assert row_check.item()
        assert col_check.item()

        return similarity_score


if __name__ == "__main__":
    x = torch.randn(1000, 128)
    y = torch.randn(1000, 128)

    metric = SoftMatch()
    print(f"kfold fit: {metric.fit_kfold_score(x, y)}")
