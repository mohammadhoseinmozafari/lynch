from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from ulid import ulid

from case.domain.value_objects.profile_namespace import ProfileNamespace
from core.context import InvestigationContext


class ProfilerType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"


class DatasetProfiler(ABC):
    """Base class for profilers that inspect a pandas dataset."""

    capability: str
    requires: set[str] = set()
    provides: set[str] = set()

    def __init__(self) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type: ProfilerType = ProfilerType.DATASET

    @abstractmethod
    def profile(self, ctx: InvestigationContext) -> ProfileNamespace:
        """Calculate raw statistics from an investigation context."""
        pass

    @property
    def capability_name(self) -> str:
        return self.capability
