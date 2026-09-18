from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import  Callable, Optional




class AnalyzerType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"

class AnalysisResult:

    def __init__(self, raw, visualizer: Optional[Callable]= None) -> None:
        self._raw = raw
        self._visualizer : Optional[Callable] = visualizer
    
    @property
    def raw(self):
        return self._raw
    @property    
    def plot(self):
        if self._visualizer is not None:
            return self._visualizer(self.raw)
        raise ValueError("This type of analysis doesn't have visualization")
    

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


