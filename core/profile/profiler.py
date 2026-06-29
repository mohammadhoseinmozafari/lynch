
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

from enum import Enum


class ProfilerType (str, Enum) :
    DATASET = 'dataset'
    MODEL = 'model'
class DatasetProfiler (ABC) :
    """
    Base abstract class for all dataset profilers
    """

    id : str
    profiler_type :  ProfilerType = ProfilerType.DATASET

    
    @abstractmethod
    def profile (self, df : pd.DataFrame)  -> Any:
        pass