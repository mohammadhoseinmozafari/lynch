
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar , Dict


from core.profilers.missingness.models import ColumnMissingRateProfile, MissingnessDistributionProfile, RowsMissingRateProfile


class ProfilerType (Enum, str) :
    PANDAS = 'pandas'
    POLARS = 'polars'

DataFrameT = TypeVar("DataFrameT")

class MissingRateProfiler (ABC , Generic[DataFrameT]) :

    id : str
    profiler_type :  ProfilerType

    
    @abstractmethod
    def profile_columns (self, df : DataFrameT) -> Dict[str, ColumnMissingRateProfile]:
        pass
    
    @abstractmethod
    def profile_rows (self, df : DataFrameT , sample_size : int) -> RowsMissingRateProfile :
        pass

    @abstractmethod
    def profile_rows_distribution (self, df: DataFrameT) -> MissingnessDistributionProfile:
        pass

    @abstractmethod
    def profile_columns_distribution (self, df: DataFrameT) -> MissingnessDistributionProfile:
        pass