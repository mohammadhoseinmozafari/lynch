from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from core.shared.domain.value_objects.validation_error import ValidationError
from core.shared.domain.value_objects.validation_warning import ValidationWarning
from core.shared.domain.enums.warning_severity import WarningSeverity

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
    
