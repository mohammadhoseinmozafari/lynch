# src/domain/repositories/explanation_link_repository.py
from abc import ABC, abstractmethod
from typing import Optional
from core.case.domain.entities.explanation_link import ExplanationLink
from core.case.domain.value_objects.ids import VersionId

class ExplanationLinkRepository(ABC):
    @abstractmethod
    def insert(self, link: ExplanationLink) -> None:
        pass

    @abstractmethod
    def find_by_model_version_id(self, version_id: VersionId) -> Optional[ExplanationLink]:
        pass

    @abstractmethod
    def update_explanation_id(self, link_id: str, explanation_id: str) -> None:
        pass