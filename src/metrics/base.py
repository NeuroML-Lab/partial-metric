from typing import List
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold

import torch


class Metric(ABC):
    """
    abstract base metric class for all alignment-based metrics
    """

    def __init__(self, k_folds: int = 5):
        self.k_folds = k_folds

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
        """
        fit a distribution pair using any metric
        """
        pass

    @abstractmethod
    def score(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
        """
        compute a similarity score between a
        distribution pair from the distance metric
        """
        pass

    def fit_score(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
        """
        short hand for fit -> score (fit optimal transform ->
        compute correlation-based score)
        """
        return self.fit(x=x, y=y).score(x=x, y=y)

    def fit_kfold_score(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> List[float]:
        """
        compute optimal transformation by running k-fold cross
        validation, returning 1 score per fold.
        """
        similarity_scores = []
        # instantiate a k-fold object
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(x):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            # compute optimal transformation matrix on the train set
            self.fit(x=x_train, y=y_train)
            # and score on test set
            score = self.score(x=x_test, y=y_test, use_kfold=True)
            similarity_scores.append(score)

        return similarity_scores
