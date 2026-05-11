
from __future__ import annotations
from pydantic import BaseModel, ConfigDict,Field
from typing import Any, List, Optional
from enum import Enum
class DataSchemaErrorCode(str, Enum):
    INCONSISTENT_SCHEMA = "INCONSISTENT_SCHEMA"
    TRAIN_EMPTY = "TRAIN_EMPTY"


class DataSchemaWarningCode(str, Enum):
    VAL_EMPTY = "VAL_EMPTY"
    TEST_EMPTY = "TEST_EMPTY"
    TARGETS_EMPTY = "TARGETS_EMPTY"


class WarningSeverity(str, Enum):
    """Severity level for validation warnings."""
    LOW = "low"         # Informational, no impact
    MEDIUM = "medium"   # Reduces confidence, should fix before production
    HIGH = "high"       # Critical, blocks production deployment

class ValidationError(BaseModel):
    code : Any
    detail: str = Field (max_length=255)

class DataSchemaValidationError(ValidationError):
    feature_names: Optional[List[str]] = Field(default=None)
    code: DataSchemaErrorCode
    detail: str = Field (max_length=255)
    model_config=ConfigDict(frozen=True)

class DataSchemaValidationWarning(BaseModel):
    code: DataSchemaWarningCode
    detail: str = Field(max_length=255)
    severity: WarningSeverity
    feature_names: Optional[List[str]] = Field(default=None)
    model_config= ConfigDict(frozen=True)

class DataSchemaValidationReport(BaseModel):
    is_valid: bool = True
    errors: List[DataSchemaValidationError] = Field(default_factory=list)
    warnings: List[DataSchemaValidationWarning] = Field(default_factory=list)

    def add_error(self, code: DataSchemaErrorCode, detail: str ,feature_names: Optional[List[str]] = None) -> None:
        self.errors.append(DataSchemaValidationError(code=code, detail= detail, feature_names= feature_names))

    def add_warning(self, code: DataSchemaWarningCode, detail: str, severity:WarningSeverity, feature_names: Optional[List[str]]= None) -> None:
        self.warnings.append(DataSchemaValidationWarning(code= code, detail= detail, severity=severity, feature_names= feature_names))
