# src/domain/repositories/model_version_repository.py
from abc import ABC, abstractmethod
from typing import Optional
from core.case.domain.entities.model_version import ModelVersion
from core.case.domain.value_objects.ids import FamilyId, VersionId

class ModelVersionRepository(ABC):
    @abstractmethod
    def insert(self, version: ModelVersion) -> None:
        """Persist a new model version."""
        pass

    @abstractmethod
    def find_by_id(self, version_id: VersionId) -> Optional[ModelVersion]:
        """Retrieve a model version by its unique ID."""
        pass

    @abstractmethod
    def find_by_family_and_version(self, family_id: FamilyId, version_number: int) -> Optional[ModelVersion]:
        """Find a version by its family and semantic version number."""
        pass

    @abstractmethod
    def get_next_version_number(self, family_id: FamilyId) -> int:
        """Return the next auto‑incremented version number for a family."""
        pass