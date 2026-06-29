
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import pandas as pd



class ProfilerType (str, Enum) :
    DATASET = 'dataset'
    MODEL = 'model'


class DatasetProfiler (ABC) :

    id : str
    profiler_type :  ProfilerType = ProfilerType.DATASET

    
    @abstractmethod
    def profile (self, df : pd.DataFrame)  -> Any:
        pass