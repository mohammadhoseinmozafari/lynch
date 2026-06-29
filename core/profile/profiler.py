
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from ulid import ulid
from enum import Enum


class ProfilerType (str, Enum) :
    DATASET = 'dataset'
    MODEL = 'model'
class DatasetProfiler (ABC) :
    """
    Base abstract class for all dataset profilers
    """

    profiler_type :  ProfilerType = ProfilerType.DATASET
    def __init__(self) -> None:
        super().__init__()
        self.id = str(ulid())
    
    @abstractmethod
    def profile (self, df : pd.DataFrame)  -> Any:
        pass