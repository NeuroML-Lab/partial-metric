import ot
import numpy as np
from typing import Optional, List, Union

from src.metrics.base import Metric

import torch
from torchmetrics.functional import pairwise_cosine_similarity


class SlicedSoftMatch(Metric):
    """
    compute an optimal soft-permutation matrix after projecting
    multivariate distributions onto the real line.
    """

    def __init__(
        self, p: int = 2, num_projections: int = 50, k_folds: Optional[int] = 5
    ):
        self.p = p
        self.num_projections = num_projections
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

    @staticmethod
    def _random_orthogonal_projections(
        num_dims: int, num_projections: int
    ) -> torch.Tensor:
        """
        generate `num_projections` randomly sampled, orthogonalized
        projection matrices
        """
        # compute the number of factorizations
        num_factors = int(num_projections / num_dims) + 1
        projections = torch.zeros((num_dims * num_factors, num_dims))

        for factor in range(num_factors):
            # sample a random gaussian matrix
            random_matrix = torch.randn(num_dims, num_dims)
            # orthogalize the basis of the random matrix
            q, r = torch.linalg.qr(random_matrix)
            projections[factor * num_dims : (factor + 1) * num_dims] = q

        projections = projections[0:num_projections]

        return projections

    @staticmethod
    def _project_distribution(
        measure: torch.Tensor, projections: torch.Tensor
    ) -> torch.Tensor:
        """
        project a multivariate measure to a single dimension
        using a set of projection matrices
        """
        # unfortunately, np.dot() and torch.dot() don't
        # have the same functionality (ie: batched dot product)
        # this is a weird hacky way to overcome the issue

        return np.dot(projections, measure)
        # return projections.matmul(measure)

    def fit(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> "SlicedSoftMatch":
        """
        compute an optimal soft-permutation matrix after slicing
        multivariate measures
        """
        x = torch.tensor(x) if isinstance(x, np.ndarray) else x
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y

        # mean center responses
        self.mx_ = torch.nanmean(x.T, dim=1)[:, None]
        self.my_ = torch.nanmean(y.T, dim=1)[:, None]
        Xc = x.T - self.mx_
        Yc = y.T - self.my_

        N_x, T = Xc.shape[1], Xc.shape[0]
        N_y = Yc.shape[1]

        a = np.ones(N_x) / N_x
        b = np.ones(N_y) / N_y

        # accumulate slice distances and optionally plans
        self.slice_distances = []
        self.slice_plans = []

        for _ in range(self.num_projections):
            # random direction in stimulus space
            v = torch.randn(T, device=Xc.device)
            v = v / v.norm()
            u = (v @ Xc).cpu().numpy()
            w = (v @ Yc).cpu().numpy()

            # 1D sorting
            u_sorted = np.sort(u)
            w_sorted = np.sort(w)

            # fast 1D OT
            if self.p == 1:
                d = ot.emd_1d(u_sorted, w_sorted)
                # uniform weights so plan implicitly quantile matched
                plan = None
            else:
                d = ot.wasserstein_1d(u_sorted, w_sorted, p=self.p)
                plan = None

            self.slice_distances.append(d)
            self.slice_plans.append(plan)

        # aggregate into a single distance
        self.distance = float(np.mean(self.slice_distances))
        return self

        # x_centered = x.T - self.mx_
        # y_centered = y.T - self.my_

        ## compute cost matrix
        ## self.cost = 1 - pairwise_cosine_similarity(x_centered, y_centered)

        ## initialize uniform marginals
        # a = torch.ones(x.shape[1], device=x.device) / x.shape[1]
        # b = torch.ones(y.shape[1], device=y.device) / y.shape[1]

        # self.distance = ot.sliced_wasserstein_distance(
        #    x_centered, y_centered, a, b, n_projections=self.num_projections, p=self.p
        # )
        # project measures to 1D (in its current shape, it won't work for different dimensions)
        # projections = self._random_orthogonal_projections(
        #    num_dims=x.shape[0], num_projections=self.num_projections
        # )
        # x_projected = self._project_distribution(
        #    measure=x_centered.T, projections=projections
        # )
        # y_projected = self._project_distribution(
        #    measure=y_centered.T, projections=projections
        # )

        ## compute EMD on 1d projections (using the 2-wasserstein distance)
        # self.transform = ot.emd_1d(x_projected, y_projected, a, b, p=2)

        return self

    def score(
        self, x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor], **kwargs
    ) -> float:
        use_kfold = kwargs.get("use_kfold", False)
        x = torch.tensor(x) if isinstance(x, np.ndarray) else x
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y

        if not use_kfold:
            x = x.T - self.mx_
            y = y.T - self.my_

            # similarity_score = 1 - torch.nansum(self.transform * self.cost).item()
        else:
            raise NotImplementedError

        # return similarity_score


if __name__ == "__main__":
    x = torch.randn(100, 20)
    y = torch.randn(100, 20)

    from src.metrics.alignment.soft_match import SoftMatch

    m1 = SlicedSoftMatch(num_projections=1000, p=2)
    m2 = SoftMatch()
    m1.fit(x=x, y=y)
    m2.fit(x=x, y=y)
    sm_distance = torch.nansum(m2.transform * m2.cost).item()
    print(f"sliced distance: {m1.distance}")
    print(f"sm score: {sm_distance}")
