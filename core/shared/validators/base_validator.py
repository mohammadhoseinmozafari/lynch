from abc import ABC, abstractmethod
from graphlib import TopologicalSorter
from typing import Dict, List, Optional, Generic, TypeVar, Set
from core.shared.domain.value_objects.validation_rule import ValidationRule
from core.shared.domain.enums.validation_status import ValidationStatus
from core.shared.domain.value_objects.validation_report import ValidationReport
from core.shared.domain.enums.rule_severity import RuleSeverity
from core.shared.domain.enums.warning_severity import WarningSeverity
T = TypeVar('T')

class BaseValidator(ABC, Generic[T]):
    """
    Abstract base class for all validators.
    Validators ADD errors/warnings to report, they don't raise them.
    Exceptions are only for actual code crashes.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.rules: Dict[str, ValidationRule] = {}
        self._status: Dict[str, ValidationStatus] = {}
        self._register_rules()
    
    @abstractmethod
    def _register_rules(self) -> None:
        """Register all validation rules for this validator."""
        pass
    
    @abstractmethod
    def _get_target_id(self, target: T) -> Optional[str]:
        """Extract unique identifier from target for reporting."""
        pass
    
    def register_rule(self, rule: ValidationRule) -> None:
        """Register a validation rule."""
        self.rules[rule.name] = rule
        self._status[rule.name] = ValidationStatus.PENDING
    
    def _get_execution_order(self) -> List[str]:
        """Get topological order of rules based on dependencies."""
        dependency_graph = {
            rule_name: rule.dependencies 
            for rule_name, rule in self.rules.items()
        }
        sorter = TopologicalSorter(dependency_graph)
        return list(sorter.static_order())
    
    def _should_run_rule(self, rule_name: str, failed_blocking_rules: Set[str]) -> bool:
        """
        Determine if a rule should run based on dependency status.
        Returns False if any blocking dependency failed (added errors).
        """
        rule = self.rules[rule_name]
        
        for dep in rule.dependencies:
            if dep in failed_blocking_rules:
                return False
        
        return True
    
    def _has_dependents(self, rule_name: str) -> bool:
        """Check if any other rules depend on this one."""
        for rule in self.rules.values():
            if rule_name in rule.dependencies:
                return True
        return False
    
    def _rule_has_errors(self, report: ValidationReport, 
                         errors_before: int) -> bool:
        """
        Check if this rule added any errors or warnings.
        Compares report state before and after rule execution.
        """
        return report.error_count > errors_before 
    
    # ========================================================================
    # Main Validation Entry Point
    # ========================================================================
    
    def validate(self, target: T, report: Optional[ValidationReport] = None) -> ValidationReport:
        """
        Validate the target object.
        Rules execute in dependency order. 
        Stops early if a BLOCKING rule adds errors and has dependents.
        
        IMPORTANT: Validators ADD errors to report, they don't raise them.
        Exceptions are only caught for actual code crashes.
        """
        if report is None:
            report = ValidationReport(validator_name=self.name)
        
        # Reset status for this validation run
        for rule_name in self.rules:
            self._status[rule_name] = ValidationStatus.PENDING
        
        # Get execution order based on dependencies
        execution_order = self._get_execution_order()
        
        # Track which blocking rules have failed (added errors)
        failed_blocking_rules: Set[str] = set()
        
        # Execute rules in order
        for rule_name in execution_order:
            rule = self.rules[rule_name]
            
            # Check if we should run this rule
            if not self._should_run_rule(rule_name, failed_blocking_rules):
                self._status[rule_name] = ValidationStatus.SKIPPED
                continue
            
            # Snapshot report state before running rule
            errors_before = report.error_count
            
            # Run the validation rule
            self._status[rule_name] = ValidationStatus.RUNNING
            
            try:
                # Execute the validation function
                # The function ADDS errors/warnings to report, doesn't raise them
                rule.func(target, report, rule_name)
                
                # Check if rule added any errors/warnings
                if self._rule_has_errors(report, errors_before=errors_before):
                    self._status[rule_name] = ValidationStatus.FAILED
                    
                    # Track failed blocking rules
                    if rule.severity == RuleSeverity.BLOCKING:
                        failed_blocking_rules.add(rule_name)
                else:
                    self._status[rule_name] = ValidationStatus.PASSED
                
            except Exception as e:
                # This is a CODE CRASH, not a validation failure
                # Examples: network error, file not found, bug in validation logic
                self._status[rule_name] = ValidationStatus.CRASHED
                
                # Add as error in report
                report.add_error(
                    code="VALIDATION_RULE_CRASH",
                    detail=f"Rule '{rule_name}' crashed with exception: {str(e)}",
                    context={"rule_name": rule_name, "error_type": type(e).__name__, "error": str(e)}
                )
                
                # Crashed blocking rules also stop the pipeline
                if rule.severity == RuleSeverity.BLOCKING:
                    failed_blocking_rules.add(rule_name)
            
            # Check if we should stop the pipeline
            if self._should_stop_pipeline(rule_name, failed_blocking_rules):
                # Mark remaining pending rules as skipped
                for remaining_rule in execution_order[execution_order.index(rule_name) + 1:]:
                    if self._status[remaining_rule] == ValidationStatus.PENDING:
                        self._status[remaining_rule] = ValidationStatus.SKIPPED
                break
        
        return report
    
    def _should_stop_pipeline(self, rule_name: str, failed_blocking_rules: Set[str]) -> bool:
        """
        Determine if pipeline should stop after this rule.
        Stops if:
        1. Rule is BLOCKING severity
        2. Rule failed (added errors) or crashed
        3. Rule has dependents (other rules need it)
        """
        rule = self.rules[rule_name]
        
        if rule.severity == RuleSeverity.BLOCKING:
            if rule_name in failed_blocking_rules:
                if self._has_dependents(rule_name):
                    return True
        
        return False
    
    def get_rule_status(self) -> Dict[str, ValidationStatus]:
        """Get status of all rules from last validation run."""
        return self._status.copy()
    
    def print_summary(self, report: ValidationReport) -> None:
        """Pretty print validation results to console."""
        print(f"\n{'='*60}")
        print(f"Validator: {report.validator_name}")
        print(f"{'='*60}")
        
        if report.has_errors:
            print(f"\n❌ ERRORS ({report.error_count}):")
            for error in report.errors:
                print(f"  [{error.code}] {error.detail}")
                if error.context:
                    print(f"     Context: {error.context}")
        
        if report.has_warnings:
            print(f"\n⚠️  WARNINGS ({report.warning_count}):")
            for warning in report.warnings:
                severity_icon = {
                    WarningSeverity.HIGH: "🔴",
                    WarningSeverity.MEDIUM: "🟡",
                    WarningSeverity.LOW: "🔵"
                }.get(warning.severity, "⚪")
                print(f"  {severity_icon} [{warning.code}] {warning.detail}")
                if warning.context:
                    print(f"     Context: {warning.context}")
        
        if not report.has_errors and not report.has_warnings:
            print("\n✅ All validation rules passed!")
        
        # Print execution stats
        print(f"\n📊 Execution Stats:")
        passed = sum(1 for s in self._status.values() if s == ValidationStatus.PASSED)
        failed = sum(1 for s in self._status.values() if s == ValidationStatus.FAILED)
        crashed = sum(1 for s in self._status.values() if s == ValidationStatus.CRASHED)
        skipped = sum(1 for s in self._status.values() if s == ValidationStatus.SKIPPED)
        print(f"  Passed: {passed}")
        print(f"  Failed (added errors): {failed}")
        print(f"  Crashed (exceptions): {crashed}")
        print(f"  Skipped: {skipped}")
        print(f"{'='*60}\n")
