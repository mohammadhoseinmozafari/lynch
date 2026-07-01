# src/domain/repositories/artifact_repository.py
from abc import ABC, abstractmethod
from typing import Optional
from core.case.domain.entities.model_artifact import ModelArtifact
from core.case.domain.value_objects.ids import VersionId

class ArtifactRepository(ABC):
    @abstractmethod
    def insert(self, artifact: ModelArtifact) -> None:
        pass

    @abstractmethod
    def find_by_model_version_id(self, version_id: VersionId) -> Optional[ModelArtifact]:
        pass

    @abstractmethod
    def update_tier(self, artifact_id: str, new_tier: str) -> None:
        pass