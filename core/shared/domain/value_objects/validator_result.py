
from core.shared.domain.enums.validation_status import ValidationStatus
from core.shared.domain.enums.validator_exec_status import ValidatorExecutionStatus
from core.shared.domain.value_objects.validation_report import ValidationReport

from typing import (
    Optional,
    Dict,

)
from pydantic import BaseModel, ConfigDict, Field

class ValidatorResult(BaseModel):
    """
    Wrapper around a validator's execution result.
    
    Contains the validation report and execution metadata.
    """
    validator_name: str
    status: ValidatorExecutionStatus
    report: Optional[ValidationReport] = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    rule_statuses: Dict[str, ValidationStatus] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    @property
    def has_errors(self) -> bool:
        return self.report is not None and self.report.has_errors
    
    @property
    def has_warnings(self) -> bool:
        return self.report is not None and self.report.has_warnings
    
    @property
    def was_successful(self) -> bool:
        return self.status == ValidatorExecutionStatus.SUCCESS
    
    @property
    def should_stop_chain(self) -> bool:
        """Determine if this result should stop the chain (for fail-fast)."""
        return not self.was_successful
