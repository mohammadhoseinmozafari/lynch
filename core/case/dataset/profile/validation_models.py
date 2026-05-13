from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional



class DataProfileErrorCode(str, Enum):
    PROFILE_SCHEMA_MISMATCH = "PROFILE_SCHEMA_MISMATCH"  # Profile Splits Must Match Schema Splits
    FEATURE_PROFILE_MISMATCH = "FEATURE_PROFILE_MISMATCH" #  Every Feature in the Schema Must Have a Profile
    FEATURE_TYPE_MISMATCH = "FEATURE_TYPE_MISMATCH" # Feature Profile Type Must Match Schema Data Type
    INCONSISTENT_NUMERIC_STATS = "INCONSISTENT_NUMERIC_STATS"
    INCONSISTENT_CATEGORICAL_STATS = "INCONSISTENT_CATEGORICAL_STATS"
    ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"
    SPLIT_FRACTION_INCONSISTENCY = "SPLIT_FRACTION_INCONSISTENCY"
    DRIFT_BASELINE_NOT_FOUND = "DRIFT_BASELINE_NOT_FOUND"
    MISSING_RATE_STATS_MISMATCH = "MISSING_RATE_STATS_MISMATCH"
    CROSS_SPLIT_TYPE_MISMATCH = "CROSS_SPLIT_TYPE_MISMATCH"



class DataProfileWarningCode(str, Enum):
    SPLIT_FRACTION_INCONSISTENCY = "SPLIT_FRACTION_INCONSISTENCY"
    MISSING_RATE_STATS_MISMATCH = "MISSING_RATE_STATS_MISMATCH"
    STATS_MISSING = "STATS_MISSING"
    HIGH_MISSING_RATE = "HIGH_MISSING_RATE"
    HIGH_CARDINALITY_INCOMPLETE_TOP_VALUES = "HIGH_CARDINALITY_INCOMPLETE_TOP_VALUES"
    ZERO_VARIANCE_FEATURE = "ZERO_VARIANCE_FEATURE"
    NEGATIVE_VARIANCE_FEATURE = "NEGATIVE_VARIANCE_FEATURE"
    LOW_TOP_VALUES_COVERAGE = "LOW_TOP_VALUES_COVERAGE"
    EMPTY_BASELINE_STATS = "EMPTY_BASELINE_STATS"
    ROW_COUNT_SLIGHT_MISMATCH = "ROW_COUNT_SLIGHT_MISMATCH"
    TOP_VALUES_MISSING = "TOP_VALUES_MISSING"
    IMBALANCED_SPLITS = "IMBALANCED_SPLITS"
    


class DataProfileValidationError(BaseModel):
    feature_name: Optional[str] = Field(default=None)
    code: DataProfileErrorCode
    detail: str
    model_config= ConfigDict(frozen=True)


class DataProfileValidationWarning(BaseModel):
    feature_name: Optional[str] = Field(default=None)
    code: DataProfileWarningCode
    detail: str
    model_config= ConfigDict(frozen=True)




class DataProfileValidationReport(BaseModel):
    is_valid: bool = True
    errors: List[DataProfileValidationError] = Field(default_factory=List)
    warnings: List[DataProfileValidationWarning] = Field(default_factory=List)

    def add_error(self, code: DataProfileErrorCode, detail: str, feature_name: Optional[str]= None):
        self.errors.append(DataProfileValidationError(feature_name= feature_name, code=code, detail= detail))
        self.is_valid = False

    def add_warning(self,  code: DataProfileWarningCode, detail: str, feature_name: Optional[str]= None):
        self.warnings.append(DataProfileValidationWarning(feature_name= feature_name, code= code, detail= detail))
