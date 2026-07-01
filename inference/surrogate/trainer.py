"""Training boundary for models used internally by Holmes."""

from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Protocol, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, clone

from inference.surrogate.models import (
    SurrogateTrainingError,
    SurrogateTrainingRequest,
    SurrogateTrainingResult,
)


EstimatorT = TypeVar("EstimatorT", bound=BaseEstimator)


class SurrogateTrainer(Protocol):
    """Port for fitting Holmes-owned, non-user models."""

    def train(
        self,
        request: SurrogateTrainingRequest[EstimatorT],
    ) -> SurrogateTrainingResult[EstimatorT]: ...


class SklearnSurrogateTrainer:
    """Fit a cloned sklearn estimator after validating the training arrays."""

    def train(
        self,
        request: SurrogateTrainingRequest[EstimatorT],
    ) -> SurrogateTrainingResult[EstimatorT]:
        features = np.asarray(request.features)
        target = np.asarray(request.target)
        self._validate(features, target, request.purpose)

        estimator = clone(request.estimator)
        started = perf_counter()
        try:
            estimator.fit(features, target)
        except Exception as exc:  # sklearn exposes estimator-specific failures.
            raise SurrogateTrainingError(
                f"Failed to train surrogate for {request.purpose!r}: {exc}"
            ) from exc
        duration = perf_counter() - started

        distribution = Counter(str(value) for value in target.tolist())
        return SurrogateTrainingResult(
            estimator=estimator,
            purpose=request.purpose,
            estimator_type=type(estimator).__name__,
            sample_count=int(features.shape[0]),
            feature_count=int(features.shape[1]),
            target_distribution=dict(sorted(distribution.items())),
            training_duration_seconds=duration,
            metadata=dict(request.metadata),
        )

    @staticmethod
    def _validate(features: np.ndarray, target: np.ndarray, purpose: str) -> None:
        if not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if target.ndim != 1:
            raise ValueError("target must be a one-dimensional array")
        if features.shape[0] == 0:
            raise ValueError("surrogate training requires at least one sample")
        if features.shape[1] == 0:
            raise ValueError("surrogate training requires at least one feature")
        if features.shape[0] != target.shape[0]:
            raise ValueError("features and target must have the same row count")
        if target.size == 0:
            raise ValueError("target must not be empty")
