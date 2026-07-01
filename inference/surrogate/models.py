"""Contracts for Holmes-internal surrogate model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np


EstimatorT = TypeVar("EstimatorT")


@dataclass(frozen=True)
class SurrogateTrainingRequest(Generic[EstimatorT]):
    """A side-effect-free request to fit an internal estimator."""

    estimator: EstimatorT
    features: np.ndarray
    target: np.ndarray
    purpose: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurrogateTrainingResult(Generic[EstimatorT]):
    estimator: EstimatorT
    purpose: str
    estimator_type: str
    sample_count: int
    feature_count: int
    target_distribution: dict[str, int]
    training_duration_seconds: float
    metadata: dict[str, Any]

    def model_metadata(self) -> dict[str, Any]:
        """Return persistence-safe metadata without exposing the estimator."""
        return {
            "purpose": self.purpose,
            "estimator_type": self.estimator_type,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "target_distribution": dict(self.target_distribution),
            "training_duration_seconds": self.training_duration_seconds,
            **self.metadata,
        }


class SurrogateTrainingError(RuntimeError):
    """Raised when an internal surrogate cannot be trained safely."""
