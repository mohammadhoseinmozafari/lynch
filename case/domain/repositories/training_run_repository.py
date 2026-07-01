# src/domain/repositories/training_run_repository.py
from abc import ABC, abstractmethod
from typing import Optional
from core.case.domain.entities.training_run import TrainingRun
from core.case.domain.value_objects.ids import VersionId

class TrainingRunRepository(ABC):
    @abstractmethod
    def insert(self, run: TrainingRun) -> None:
        pass

    @abstractmethod
    def find_by_model_version_id(self, version_id: VersionId) -> Optional[TrainingRun]:
        pass

    @abstractmethod
    def exists_by_hash(self, training_run_hash: str) -> bool:
        """Check if any training run with the given hash already exists."""
        pass