from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, List




class DatasetErrorCode(str, Enum):
    INCONSISTENT_SCHEMA = "INCONSISTENT_SCHEMA"
    TRAIN_EMPTY = "TRAIN_EMPTY"
    DRIFT_BASELINE_SPLIT_DOESNT_EXIST = "DRIFT_BASELINE_SPLIT_DOESNT_EXIST"
    FEATURE_PROFILE_MISMATCH = "FEATURE_PROFILE_MISMATCH"

class DatasetWarningCode(str, Enum):
    VAL_EMPTY = "VAL_EMPTY"
    TEST_EMPTY = "TEST_EMPTY"
    TARGETS_EMPTY = "TARGETS_EMPTY"
    TASK_TYPE_EMPTY = "TASK_TYPE_EMPTY"
    DRIFT_BASELINE_EMPTY = "DRIFT_BASELINE_EMPTY" 
    INVALID_SPLIT_FRACTIONS_SUM = "INVALID_SPLIT_FRACTIONS_SUM"
    SPLIT_FRACTIONS_NOT_PROVIDED = "SPLIT_FRACTIONS_NOT_PROVIDED"
    DATA_PROFILE_EMPTY  = "DATA_PROFILE_EMPTY"

    
class WarningSeverity(str, Enum):
    """Severity level for validation warnings."""
    LOW = "low"         # Informational, no impact
    MEDIUM = "medium"   # Reduces confidence, should fix before production
    HIGH = "high"       # Critical, blocks production deployment

class DatasetValidationError(BaseModel):
    code : Any
    detail: str = Field(max_length=255)
    feature_names : Optional[List[str]] = Field(default=None)

class DatasetValidationWarning (BaseModel):
    code : Any
    detail: str = Field(max_length=255)
    severity: WarningSeverity
    feature_names : Optional[List[str]] = Field(default= None)

class DatasetValidationReport(BaseModel):
    is_valid : bool = True
    errors: List[Any] = Field(default_factory=list)
    warnings: List[Any] = Field(default_factory=list)

    def add_error(self, code: DatasetErrorCode, detail: str ,feature_names: Optional[List[str]] = None) -> None:
        self.errors.append(DatasetValidationError(code=code, detail= detail, feature_names= feature_names))

    def add_warning(self, code: DatasetWarningCode, detail: str, severity:WarningSeverity, feature_names: Optional[List[str]]= None) -> None:
        self.warnings.append(DatasetValidationWarning(code= code, detail= detail, severity=severity, feature_names= feature_names))
