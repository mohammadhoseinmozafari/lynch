from datetime import datetime, timezone
import traceback

from pydantic import (
    BaseModel, 
    Field
)

from typing import (
    Dict,
    Generic,
    List,
    TypeVar, 
    Any,
    Optional
)
import logging

from core.shared.domain.enums.validation_status import ValidationStatus
from core.shared.domain.enums.validator_exec_status import ValidatorExecutionStatus
from core.shared.domain.value_objects.aggregated_validation import AggregatedValidation
from core.shared.domain.value_objects.validator_result import ValidatorResult
from core.shared.validators import ValidatorAdapter
from core.shared.domain.value_objects.validation_report import ValidationReport

logger = logging.getLogger(__name__)


R = TypeVar('R')

class ValidatorChainConfig(BaseModel):
    """Configuration for the synchronous ValidatorChain."""
    # Execution
    fail_fast: bool = False                  # Stop on first error
    
    # Error handling
    continue_on_validator_crash: bool = True  # Don't stop chain if a validator crashes
    treat_crashes_as_errors: bool = True      # Crashes produce blocking errors
    
    # Reporting
    collect_warnings_always: bool = True      # Always collect warnings
    
    # Timing
    timeout_seconds: Optional[float] = Field(default=None, ge=0.0)


class ValidatorChain(Generic[R]):
    """
    Synchronous, type-safe ValidatorChain that works with heterogeneous validators.
    
    Generic over R (request type), not individual validator types.
    Each validator adapter knows how to extract its target from R.
    
    Execution Model:
    - Sequential: Validators run one after another in registration order
    - Synchronous: No async/await, simple function calls
    - Fail-fast: Optional early termination on first error
    - Predictable: Deterministic execution order
    
    Design:
    - Type-safe: Generic over R, adapters handle type conversion
    - Heterogeneous: Validators can validate different types
    - Composable: Validators can be added/removed dynamically
    - Simple: No threading, no async, easy to debug
    
    Usage:
        chain = ValidatorChain[RegisterModelRequest](
            ValidatorAdapter(
                SchemaExplainabilityValidator(),
                lambda req: req.dataset_binding.schema_snapshot,
                name="schema_validator"
            ),
            ValidatorAdapter(
                ProfileConsistencyValidator(),
                lambda req: req.dataset_binding.profile_snapshot,
                name="profile_validator"
            ),
            ValidatorAdapter(
                EnvironmentValidator(),
                lambda req: req.environment_snapshot,
                name="env_validator"
            )
        )
        
        result = chain.run(request)
        
        if not result.is_valid:
            raise Error
    """
    
    def __init__(
        self,
        *adapters: ValidatorAdapter[Any, R],
        config: Optional[ValidatorChainConfig] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the chain with adapters.
        
        Args:
            *adapters: One or more ValidatorAdapter instances
            config: Optional configuration
            name: Optional name for this chain (for logging)
            
        Raises:
            ValueError: If no adapters provided or duplicate names
        """
        if not adapters:
            raise ValueError("At least one validator adapter is required")
        
        self._adapters: List[ValidatorAdapter[Any, R]] = list(adapters)
        self._config = config or ValidatorChainConfig()
        self._name = name or "ValidatorChain"
        
        self._validate_adapters()
        
        # Execution state
        self._results: Dict[str, ValidatorResult] = {}
    

    def _validate_adapters(self):
        """Ensure no duplicate adapter names and validate configuration."""
        names = [a.name for a in self._adapters]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate validator names detected: {duplicates}. "
                f"Each validator must have a unique name."
            )
    
    @property
    def registered_validators(self) -> List[str]:
        """Returns list of registered validator names."""
        return [a.name for a in self._adapters]
    
   
    @property
    def last_results(self) -> Dict[str, ValidatorResult]:
        """Get results from the last execution."""
        return self._results.copy()
 

    # ========================================================================
    # Main Entry Point
    # ========================================================================
    
    def run(
        self, 
        request: R,
    ) -> AggregatedValidation:
        """
        Execute all validators sequentially against the request.
        
        This is the main entry point. Validators execute in registration order.
        
        Args:
            request: The full request object (type R)
            
        Returns:
            AggregatedValidation combining all results
            
        Raises:
            Never raises - all errors are captured in the result
            
        Behavior:
            - Extracts targets from request for each validator
            - Runs validators sequentially
            - Handles extraction failures gracefully
            - Aggregates all results
            - Optionally stops early on fail-fast
        """
        chain_start = datetime.now(timezone.utc)
        
        
        logger.info(
            f"Starting validation chain '{self._name}' "
            f"validators={self.registered_validators}"
        )
        
        # Execute all validators sequentially
        results = self._execute_all(request)
        
        execution_time = (datetime.now(timezone.utc) - chain_start).total_seconds() * 1000
        
        # Store results
        self._results = {}
        for result in results:
            self._results[result.validator_name] = result
        
        # Aggregate
        aggregated = self._aggregate(results, execution_time)
        
        logger.info(
            f"Validation chain '{self._name}' complete "
            f"valid={aggregated.is_valid} "
            f"errors={aggregated.error_count} "
            f"warnings={aggregated.warning_count} "
            f"time={execution_time:.2f}ms"
        )
        
        return aggregated
    
    
    # ========================================================================
    # Sequential Execution
    # ========================================================================
    
    def _execute_all(
        self, 
        request: R, 
    ) -> List[ValidatorResult]:
        """
        Execute all validators sequentially.
        
        Validators run in registration order. If fail_fast is enabled,
        execution stops after the first failure.
        
        Args:
            request: The full request object
            
        Returns:
            List of ValidatorResult for each validator
        """
        results = []
        
        for adapter in self._adapters:
            logger.debug(
                f"Executing validator '{adapter.name}' "
                f"position={len(results) + 1}/{len(self._adapters)}"
            )
            
            # Execute the adapter
            result = self._execute_adapter(adapter, request)
            results.append(result)
            
            # Log result
            logger.debug(
                f"Validator '{adapter.name}' complete: "
                f"status={result.status.value} "
                f"time={result.execution_time_ms:.2f}ms"
            )
            
            # Check if we should stop (fail-fast)
            if self._should_stop_after(result):
                logger.info(
                    f"Fail-fast triggered after validator '{adapter.name}' "
                )
                # Mark remaining validators as skipped
                for remaining in self._adapters[len(results):]:
                    results.append(ValidatorResult(
                        validator_name=remaining.name,
                        status=ValidatorExecutionStatus.SKIPPED
                    ))
                break
        
        return results
    
    def _should_stop_after(self, result: ValidatorResult) -> bool:
        """
        Determine if the chain should stop after this result.
        
        Stops if:
        - fail_fast is enabled AND
        - result indicates failure (not successful)
        """
        return self._config.fail_fast and result.should_stop_chain
    
    # ========================================================================
    # Single Adapter Execution
    # ========================================================================
    
    def _execute_adapter(
        self, 
        adapter: ValidatorAdapter[Any, R],
        request: R,
    ) -> ValidatorResult:
        """
        Execute a single adapter:
        1. Extract target from request
        2. Validate the target
        3. Handle any errors
        
        This is the core execution method. All errors are captured
        and returned as results, never raised.
        
        Args:
            adapter: The validator adapter to execute
            request: The full request object
            
        Returns:
            ValidatorResult with execution details
        """
        start = datetime.now(timezone.utc)
        
        try:
            # Step 1: Extract target from request
            target = adapter.extract_target(request)
            
            # Step 2: Validate the target
            report = adapter.validate(target)
            
            execution_time = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            
            # Determine status based on report
            if report.has_errors:
                status = ValidatorExecutionStatus.FAILED
            else:
                status = ValidatorExecutionStatus.SUCCESS
            
            return ValidatorResult(
                validator_name=adapter.name,
                status=status,
                report=report,
                execution_time_ms=execution_time,
                rule_statuses=adapter.get_rule_status()
            )
            
            
        except Exception as e:
            # Unexpected crash during validation
            execution_time = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            
            logger.error(
                f"Validator '{adapter.name}' crashed: {e}",
                exc_info=True,
            )
            
            return self._create_crash_result(
                adapter, e, execution_time
            )
    
    def _create_crash_result(
        self, 
        adapter: ValidatorAdapter[Any, R],
        error: Exception,
        execution_time_ms: float
    ) -> ValidatorResult:
        """Create result for validator crash."""
        report = ValidationReport(validator_name=adapter.name)
        
        if self._config.treat_crashes_as_errors:
            report.add_error(
                code="VALIDATOR_CRASH",
                detail=f"Validator '{adapter.name}' crashed: {str(error)}",
                context={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "validator": adapter.name,
                    "description": adapter.description
                }
            )
        
        return ValidatorResult(
            validator_name=adapter.name,
            status=ValidatorExecutionStatus.CRASHED,
            report=report,
            error=error,
            execution_time_ms=execution_time_ms
        )
    
    # ========================================================================
    # Aggregation (UC-01 Step 4.3)
    # ========================================================================
    
    def _aggregate(
        self, 
        results: List[ValidatorResult],
        execution_time_ms: float,
    ) -> AggregatedValidation:
        """
        Aggregate multiple validator results into a single result.
        
        Implements UC-01 Step 4.3:
        - Collect all errors from all reports
        - If any report has errors, set is_valid = False
        - Collect all warnings from all reports
        
        Args:
            results: List of validator execution results
            execution_time_ms: Total chain execution time
            
        Returns:
            AggregatedValidation combining all results
        """
        blocking_errors = []
        
        for result in results:
            if result.report is None:
                continue
            
            if result.report.has_errors:
                for error in result.report.errors:
                    blocking_errors.append({
                        "validator": result.validator_name,
                        "code": error.code,
                        "detail": error.detail,
                        "context": error.context,
                        "validator_status": result.status.value
                    })
        
        is_valid = len(blocking_errors) == 0
        
        if is_valid:
            return AggregatedValidation.create_success(
                results, execution_time_ms
            )
        else:
            return AggregatedValidation.create_failure(
                results, blocking_errors, execution_time_ms
            )
    

    def print_summary(self, aggregated: Optional[AggregatedValidation] = None) -> None:
        """Print a human-readable summary of the validation chain."""
        if aggregated is None and not self._results:
            print("No validation results available. Run the chain first.")
            return
        
        if aggregated is None:
            # Build from last results
            results = list(self._results.values())
            aggregated = self._aggregate(
                results, 
                sum(r.execution_time_ms for r in results),
            )
        
        print(f"\n{'='*70}")
        print(f"VALIDATOR CHAIN: {self._name}")
        print(f"{'='*70}")
        
        if aggregated.is_valid:
            print(f"\n✅ ALL VALIDATIONS PASSED")
        else:
            print(f"\n❌ VALIDATION FAILED")
        
        print(f"Total Time: {aggregated.execution_time_ms:.2f}ms")
        print(f"Validators Run: {aggregated.report_count}")
        print(f"Total Errors: {aggregated.error_count}")
        print(f"Total Warnings: {aggregated.warning_count}")
        
        # Execution summary
        summary = aggregated.execution_summary
        print(f"\n{'─'*70}")
        print(f"EXECUTION SUMMARY:")
        print(f"  ✅ Successful: {summary.get('successful', 0)}")
        print(f"  ❌ Failed: {summary.get('failed', 0)}")
        print(f"  💥 Crashed: {summary.get('crashed', 0)}")
        print(f"  ⏭️  Skipped: {summary.get('skipped', 0)}")
        
        # Per-validator details
        print(f"\n{'─'*70}")
        print(f"PER-VALIDATOR DETAILS:")
        print(f"{'─'*70}")
        
        for result in aggregated.validator_results:
            status_icon = {
                ValidatorExecutionStatus.SUCCESS: "✅",
                ValidatorExecutionStatus.FAILED: "❌",
                ValidatorExecutionStatus.CRASHED: "💥",
                ValidatorExecutionStatus.SKIPPED: "⏭️",
                ValidatorExecutionStatus.NOT_RUN: "⏸️"
            }.get(result.status, "❓")
            
            print(f"\n{status_icon} {result.validator_name}")
            print(f"   Status: {result.status.value}")
            print(f"   Time: {result.execution_time_ms:.2f}ms")
            
            if result.report:
                print(f"   Errors: {result.report.error_count}")
                print(f"   Warnings: {result.report.warning_count}")
            
            if result.rule_statuses:
                passed = sum(
                    1 for s in result.rule_statuses.values() 
                    if s == ValidationStatus.PASSED
                )
                failed = sum(
                    1 for s in result.rule_statuses.values() 
                    if s == ValidationStatus.FAILED
                )
                crashed = sum(
                    1 for s in result.rule_statuses.values() 
                    if s == ValidationStatus.CRASHED
                )
                skipped = sum(
                    1 for s in result.rule_statuses.values() 
                    if s == ValidationStatus.SKIPPED
                )
                print(f"   Rules: {passed}P/{failed}F/{crashed}C/{skipped}S")
        
        # Blocking errors
        if aggregated.blocking_errors:
            print(f"\n{'─'*70}")
            print(f"BLOCKING ERRORS ({len(aggregated.blocking_errors)}):")
            print(f"{'─'*70}")
            for error in aggregated.blocking_errors:
                print(f"\n  Validator: {error['validator']}")
                print(f"  Code: {error['code']}")
                print(f"  Detail: {error['detail']}")