from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Dict, List


@dataclass
class ColumnMissingRateProfile:
    column_name : str
    missing_rate : float
    missing_count : int
    non_missing_count : int
    total_count : int
    

    def to_dict(self) -> Dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class RowsMissingRateProfile:
    
    total_rows : int

    full_missing_rows_count : int
    full_missing_rows_rate : float
    full_missing_rows_sample_indices : List[int]

    high_missing_rate_rows_count : int
    high_missing_rate_rows_rate : float
    high_missing_rate_rows_sample_indices : List[int]

    sample_size : int

    def to_dict(self) -> Dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class MissingnessDistributionProfile:
    mean_missing_rate : float
    median_missing_rate : float
    std_missing_rate : float

    min_missing_rate : float
    max_missing_rate : float

    p90_missing_rate : float
    p95_missing_rate : float
    p99_missing_rate : float

    def to_dict(self) -> Dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class SegmentProfile:
    rules: List[str]

    row_count: int
    coverage: float

    missing_count: int
    missing_rate: float

    global_missing_rate: float

    contribution: float

    lift: float

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
