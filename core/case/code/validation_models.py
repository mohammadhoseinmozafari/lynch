"""
Code snapshot domain validator.

Validates that a CodeSnapshot (already structurally validated by Pydantic)
meets reproducibility requirements for explainability trust.

This is a CUSTOM validator — it handles domain rules that Pydantic cannot:
  - Repo reachability checks (network I/O)
  - Commit existence verification
  - Uncommitted diff size and format validation
  - Reproducibility classification (COMPLETE, PARTIAL, MINIMAL)
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
# =============================================================================
# Enums
# =============================================================================


class CodeCompleteness(str, Enum):
    """
    Reproducibility classification for a code snapshot.
    
    Values are ordered from most to least reproducible.
    """
    COMPLETE = "complete"       # Fully reproducible from git
    PARTIAL = "partial"         # Reproducible with caveats
    MINIMAL = "minimal"         # Limited reproducibility
    UNKNOWN = "unknown"         # Could not verify


class WarningSeverity(str, Enum):
    """Severity level for validation warnings."""
    LOW = "low"         # Informational, no impact
    MEDIUM = "medium"   # Reduces confidence, should fix before production
    HIGH = "high"       # Critical, blocks production deployment


# =============================================================================
# Error Codes
# =============================================================================


class CodeSnapshotErrorCode(str, Enum):
    """Machine-readable error codes for code snapshot validation failures."""
    INVALID_REPO_URL = "INVALID_REPO_URL"
    INVALID_COMMIT_HASH = "INVALID_COMMIT_HASH"
    MISSING_ENTRY_POINT = "MISSING_ENTRY_POINT"


class CodeSnapshotWarningCode(str, Enum):
    """Machine-readable warning codes for code snapshot validation."""
    LOCAL_REPO_NO_GIT = "LOCAL_REPO_NO_GIT"
    UNCOMMITTED_CHANGES = "UNCOMMITTED_CHANGES"
    BRANCH_NOT_TAG = "BRANCH_NOT_TAG"
    DIFF_TRUNCATED = "DIFF_TRUNCATED"
    DIFF_NOT_PARSEABLE = "DIFF_NOT_PARSEABLE"
    REPO_NOT_REACHABLE = "REPO_NOT_REACHABLE"
    COMMIT_NOT_FOUND = "COMMIT_NOT_FOUND"
    COMMIT_UNVERIFIED = "COMMIT_UNVERIFIED"
    ENTRY_POINT_NOTEBOOK = "ENTRY_POINT_NOTEBOOK"


# =============================================================================
# Error and Warning Value Objects
# =============================================================================


class CodeSnapshotError(BaseModel):
    """
    A blocking validation error for a CodeSnapshot field.
    
    Frozen (immutable) so errors can be safely collected and passed around.
    """
    field: str
    code: CodeSnapshotErrorCode
    message: str
    model_config = ConfigDict(frozen=True)
    
    def __repr__(self) -> str:
        return f"CodeSnapshotError(field={self.field!r}, code={self.code.value})"


class CodeSnapshotWarning(BaseModel):
    """
    A non-blocking validation warning for a CodeSnapshot field.
    
    Includes severity to guide deployment decisions.
    """
    field: str
    code: CodeSnapshotWarningCode
    message: str
    severity: WarningSeverity
    model_config= ConfigDict(frozen=True)
    def __repr__(self) -> str:
        return (
            f"CodeSnapshotWarning("
            f"field={self.field!r}, code={self.code.value}, severity={self.severity.value})"
        )


# =============================================================================
# Validation Report (the main output)
# =============================================================================


class CodeSnapshotValidation(BaseModel):
    """
    Complete result of validating a CodeSnapshot for reproducibility.
    
    This is the value object returned by CodeSnapshotValidator.validate().
    """
    is_valid: bool
    errors: list[CodeSnapshotError] = Field(default_factory=list)
    warnings: list[CodeSnapshotWarning] = Field(default_factory=list)
    completeness: CodeCompleteness = CodeCompleteness.UNKNOWN
    computed_hash: Optional[str] = None
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    @property
    def blocking_warnings(self) -> list[CodeSnapshotWarning]:
        """Warnings with HIGH severity that block production deployment."""
        return [w for w in self.warnings if w.severity == WarningSeverity.HIGH]
    
    @property
    def is_production_ready(self) -> bool:
        """A snapshot is production-ready if valid and has no HIGH severity warnings."""
        return self.is_valid and len(self.blocking_warnings) == 0
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def add_error(self, field: str, code: CodeSnapshotErrorCode, message: str) -> None:
        """Add a blocking error and mark the report as invalid."""
        self.errors.append(CodeSnapshotError(field=field, code=code, message=message))
        self.is_valid = False
    
    def add_warning(
        self, field: str, code: CodeSnapshotWarningCode, message: str, severity: WarningSeverity
    ) -> None:
        """Add a non-blocking warning."""
        self.warnings.append(
            CodeSnapshotWarning(field=field, code=code, message=message, severity=severity)
        )
    
    def to_dict(self) -> dict:
        """Serialize to a dictionary for JSON responses or database storage."""
        return {
            "is_valid": self.is_valid,
            "errors": [
                {"field": e.field, "code": e.code.value, "message": e.message}
                for e in self.errors
            ],
            "warnings": [
                {
                    "field": w.field,
                    "code": w.code.value,
                    "message": w.message,
                    "severity": w.severity.value,
                }
                for w in self.warnings
            ],
            "completeness": self.completeness.value,
            "computed_hash": self.computed_hash,
        }