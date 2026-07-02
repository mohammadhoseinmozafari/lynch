from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from ulid import ulid

from case.domain.value_objects.data_profile import DataProfile


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
    def profile(self, df: pd.DataFrame, profile: DataProfile) -> None:
        pass


    @property
    def capability_name(self) -> str:
       return self.capability


