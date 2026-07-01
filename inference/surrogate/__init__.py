"""Public ISS contracts for internal surrogate training."""

from inference.surrogate.models import (
    SurrogateTrainingError,
    SurrogateTrainingRequest,
    SurrogateTrainingResult,
)
from inference.surrogate.trainer import SklearnSurrogateTrainer, SurrogateTrainer

__all__ = [
    "SklearnSurrogateTrainer",
    "SurrogateTrainer",
    "SurrogateTrainingError",
    "SurrogateTrainingRequest",
    "SurrogateTrainingResult",
]
