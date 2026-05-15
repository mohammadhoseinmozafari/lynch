from typing import (
    List ,
    Dict,
    Any,
    Optional,

)
from pydantic import (
    BaseModel,
    Field,
    ConfigDict
)
from core.shared.domain.value_objects.validation_report import ValidationReport
from core.shared.domain.value_objects.validator_result import ValidatorResult
from core.shared.domain.enums.validator_exec_status import ValidatorExecutionStatus


class AggregatedValidation(BaseModel):
    """
    Aggregates results from multiple validators.
    
    Immutable Value Object - created once all validators complete.
    """
    is_valid: bool
    reports: List[ValidationReport]
    validator_results: List[ValidatorResult]
    blocking_errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    execution_summary: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    correlation_id: Optional[str] = None

    model_config= ConfigDict(frozen=True)
    
    @property
    def error_count(self) -> int:
        return len(self.blocking_errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    @property
    def report_count(self) -> int:
        return len(self.reports)
    
    @property
    def validator_names(self) -> List[str]:
        return [r.validator_name for r in self.validator_results]
    
    @classmethod
    def create_success(
        cls, 
        results: List[ValidatorResult],
        execution_time_ms: float,
        correlation_id: Optional[str] = None
    ) -> 'AggregatedValidation':
        """Factory for successful validations (all passed)."""
        reports = [r.report for r in results if r.report is not None]
        warnings = cls._extract_warnings(results)
        summary = cls._build_summary(results)
        
        return cls(
            is_valid=True,
            reports=reports,
            validator_results=results,
            blocking_errors=[],
            warnings=warnings,
            execution_summary=summary,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id
        )
    
    @classmethod
    def create_failure(
        cls, 
        results: List[ValidatorResult],
        errors: List[Dict[str, Any]],
        execution_time_ms: float,
        correlation_id: Optional[str] = None
    ) -> 'AggregatedValidation':
        """Factory for failed validations."""
        reports = [r.report for r in results if r.report is not None]
        warnings = cls._extract_warnings(results)
        summary = cls._build_summary(results)
        
        return cls(
            is_valid=False,
            reports=reports,
            validator_results=results,
            blocking_errors=errors,
            warnings=warnings,
            execution_summary=summary,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id
        )
    
    @staticmethod
    def _extract_warnings(results: List[ValidatorResult]) -> List[Dict[str, Any]]:
        """Extract all warnings from all validator results."""
        all_warnings = []
        for result in results:
            if result.report is None:
                continue
            
            for warning in result.report.warnings:
                all_warnings.append({
                    "validator": result.validator_name,
                    "code": warning.code,
                    "detail": warning.detail,
                    "severity": (
                        warning.severity.value 
                        if hasattr(warning.severity, 'value') 
                        else str(warning.severity)
                    ),
                    "context": warning.context
                })
        return all_warnings
    
    @staticmethod
    def _build_summary(results: List[ValidatorResult]) -> Dict[str, Any]:
        """Build execution summary across all validators."""
        return {
            "total_validators": len(results),
            "successful": sum(1 for r in results if r.status == ValidatorExecutionStatus.SUCCESS),
            "failed": sum(1 for r in results if r.status == ValidatorExecutionStatus.FAILED),
            "crashed": sum(1 for r in results if r.status == ValidatorExecutionStatus.CRASHED),
            "skipped": sum(1 for r in results if r.status == ValidatorExecutionStatus.SKIPPED),
            "validators_with_errors": sum(1 for r in results if r.has_errors),
            "validators_with_warnings": sum(1 for r in results if r.has_warnings),
            "validators_clean": sum(
                1 for r in results 
                if r.status == ValidatorExecutionStatus.SUCCESS 
                and not r.has_warnings
            )
        }
    
    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "validator_count": self.report_count,
            "blocking_errors": self.blocking_errors,
            "warnings": self.warnings,
            "execution_summary": self.execution_summary,
            "execution_time_ms": self.execution_time_ms,
            "correlation_id": self.correlation_id,
            "validator_details": [
                {
                    "validator_name": r.validator_name,
                    "status": r.status.value,
                    "has_errors": r.has_errors,
                    "has_warnings": r.has_warnings,
                    "execution_time_ms": r.execution_time_ms
                }
                for r in self.validator_results
            ]
        }
    
    def get_report(self, validator_name: str) -> Optional[ValidationReport]:
        """Get the report for a specific validator."""
        for result in self.validator_results:
            if result.validator_name == validator_name:
                return result.report
        return None
    
    def get_result(self, validator_name: str) -> Optional[ValidatorResult]:
        """Get the result for a specific validator."""
        for result in self.validator_results:
            if result.validator_name == validator_name:
                return result
        return None
    
    def has_errors_for(self, validator_name: str) -> bool:
        """Check if a specific validator has errors."""
        result = self.get_result(validator_name)
        return result.has_errors if result else False
    
    def has_warnings_for(self, validator_name: str) -> bool:
        """Check if a specific validator has warnings."""
        result = self.get_result(validator_name)
        return result.has_warnings if result else False