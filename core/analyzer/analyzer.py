from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


from case.domain.value_objects.profile_namespace import ProfileNamespace


class AnalyzerType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"


class Analyzer(ABC):
    """Base class for Analyzers"""

    capability: str
    requires: set[str] = set()
    provides: set[str] = set()
    analyzer_type : AnalyzerType
    id : str


    @abstractmethod
    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        """Performs analysis on models/datasets."""
        pass

    @property
    def capability_name(self) -> str:
        return self.capability


