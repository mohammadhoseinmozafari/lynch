

from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class WarningSeverity(str, Enum):
    """Severity level for validation warnings."""
    LOW = "low"         # Informational, no impact
    MEDIUM = "medium"   # Reduces confidence, should fix before production
    HIGH = "high"       # Critical, blocks production deployment


class ValidationError(BaseModel):
    """
    A blocking validation error.
    Frozen (immutable) so errors can be safely collected and passed around.
    """
    code: Any
    detail: str
    context : Optional[Dict[str, Any]] = Field(None)
    model_config= ConfigDict(frozen=True)

class ValidationWarning(BaseModel):
    """
    A non-blocking validation warning for a CodeSnapshot field.
    Includes severity to guide deployment decisions.
    """
    code : Any
    detail : str
    severity : WarningSeverity
    context : Optional[Dict[str, Any]] = Field(None)
    model_config= ConfigDict(frozen=True)

class ValidationReport (BaseModel):
    """
    Complete result of validation.
    This is the value object returned by Validator.validate().
    """
    validator_name : str
    errors : List[ValidationError] = Field(default_factory=list)
    warnings : List[ValidationWarning] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self)-> bool:
        return len(self.warnings)>0
    
    @property
    def error_count (self) -> int:
        return len (self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, 
                  code: Any, 
                  detail: str, 
                  context : Optional[Dict[str, Any] ]= None) -> None:
        
        self.errors.append(
            ValidationError(code = code, detail = detail, context = context)
        )
    
    def add_warning(self, 
                    code: Any, 
                    detail: str, 
                    severity : WarningSeverity, 
                    context : Optional[Dict[str, Any]] = None ) -> None:
        
        self.warnings.append(
            ValidationWarning(code = code, detail = detail, severity=severity ,context = context)
        )
    

class RuleSeverity (str, Enum):
    BLOCKING = "BLOCKING"
    NON_BLOCKING = "NON_BLOCKING"

class ValidationRule(BaseModel):
    """Unified rule definition"""
    name: str
    func: Callable
    dependencies: Set[str] = Field(default_factory=set)
    severity: RuleSeverity = RuleSeverity.BLOCKING 
    description: str = ""

class ValidationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CRASHED = "CRASHED"