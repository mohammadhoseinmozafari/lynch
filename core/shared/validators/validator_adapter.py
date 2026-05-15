from core.shared.domain.enums.validation_status import ValidationStatus
from core.shared.domain.value_objects.validation_report import ValidationReport
from core.shared.validators import BaseValidator
from typing import (
    Generic, 
    TypeVar, 
    Callable, 
    Optional, 
    Dict, 
    Any
)


T = TypeVar('T')
R = TypeVar('R')

class ValidatorAdapter(Generic[T, R]):
    """
    Adapts a BaseValidator[T] to work with a request of type R.
    
    This is the KEY to type safety. It bridges the gap between:
    - The validator which knows how to validate type T
    - The chain which receives requests of type R
    
    The adapter uses a target_extractor function to pull the correct
    target from the request for its validator.
    
    Example:
        # Schema validator validates DatasetSchema
        schema_adapter = ValidatorAdapter[DatasetSchema, RegisterModelRequest](
            validator=SchemaExplainabilityValidator(),
            target_extractor=lambda req: req.dataset_binding.schema_snapshot,
            name="SchemaValidator"
        )
    """
    
    def __init__(
        self,
        validator: BaseValidator[T],
        target_extractor: Callable[[R], T],
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        required: bool = True
    ):
        """
        Initialize the adapter.
        
        Args:
            validator: The BaseValidator that validates type T
            target_extractor: Function that extracts T from the request R
            name: Optional override name (defaults to validator.name)
            description: Human-readable description of what this validates
            metadata: Additional metadata for observability
            required: If True, extraction failure causes an error
        """
        self.validator = validator
        self.target_extractor = target_extractor
        self.name = name or validator.name
        self.description = description or f"Validates {validator.name}"
        self.metadata = metadata or {}
        self.required = required
        
        # Validate the extractor
        if not callable(target_extractor):
            raise ValueError(f"target_extractor for '{self.name}' must be callable")
    
    def extract_target(self, request: R) -> T:
        """
        Extract the validation target from the request.
        
        This is called by the chain before validation.
        If extraction fails, it raises a TargetExtractionError
        which the chain handles gracefully.
        
        Args:
            request: The full request object
            
        Returns:
            The extracted target of type T
            
        Raises:
            TargetExtractionError: If extraction fails
        """
        try:
            target = self.target_extractor(request)
            if target is None:
                raise TargetExtractionError(
                    f"target_extractor for '{self.name}' returned None"
                )
            return target
        except TargetExtractionError:
            raise
        except Exception as e:
            raise TargetExtractionError(
                f"Failed to extract target for '{self.name}': {str(e)}"
            ) from e
    
    def validate(
        self, 
        target: T, 
        report: Optional[ValidationReport] = None
    ) -> ValidationReport:
        """
        Validate the extracted target.
        
        Args:
            target: The extracted target of type T
            report: Optional existing report to add to
            
        Returns:
            ValidationReport with results
        """
        return self.validator.validate(target, report)
    
    def get_rule_status(self) -> Dict[str, ValidationStatus]:
        """Get rule execution status from the validator."""
        return self.validator.get_rule_status()
    
    def print_summary(self, report: ValidationReport) -> None:
        """Delegate to validator's print_summary."""
        self.validator.print_summary(report)
    
    def __repr__(self) -> str:
        return (
            f"ValidatorAdapter(name='{self.name}', "
            f"validator_type={type(self.validator).__name__}, "
            f"description='{self.description}', "
            f"required={self.required})"
        )


class TargetExtractionError(Exception):
    """Raised when target extraction from request fails."""
    pass