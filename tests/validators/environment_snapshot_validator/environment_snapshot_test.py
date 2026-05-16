
import pytest
from pathlib import Path
import json
from core.case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from core.case.domain.enums.env_snapshot_type import EnvSnapshotType
from core.case.application.validators.environment_snapshot_validator import EnvironmentValidator
from core.shared.domain.value_objects.validation_report import ValidationReport
from typing import Callable, Any

class TestEnvironmentValidator:
    """Test suite for EnvironmentValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return EnvironmentValidator()
    
    @pytest.fixture
    def validation_report(self):
        """Create empty validation report."""
        return ValidationReport()
    
    @pytest.fixture
    def load_test_case(self):
        """Load test case from JSON file."""
        def _load(file_path: str):
            with open(file_path, 'r') as f:
                return json.load(f)
        return _load
    # Test valid cases

    def test_valid_pip_lock_complete(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
        """Test complete valid PIP lock file."""
        # Arrange
        test_dir =  Path(__file__).parent / 'fixtures' / 'valid' / 'valid_pip_lock.json'
        test_case = load_test_case(test_dir)
        snapshot= EnvironmentSnapshot.model_validate(test_case)
        
        
        # Act
        report = validator.validate(snapshot, validation_report)
        assert not report.has_errors, f"Expected no errors but got: {[e.code for e in report.errors]}"
        
    def test_real_pip_freeze_output(self, validator):
        """Test with actual pip freeze output format."""
        pip_freeze_output = """
                            certifi==2023.11.17
                            charset-normalizer==3.3.2
                            idna==3.6
                            requests==2.31.0
                            urllib3==2.1.0
                            """
        snapshot = EnvironmentSnapshot(
            snapshot_type=EnvSnapshotType.PIP_LOCK,
            python_version="3.11.5",
            specification=pip_freeze_output.strip(),
            system_packages=[]
        )
        
        
        report = validator.validate(snapshot, report= None)
        
        # Should have no errors with proper pip freeze format
        assert not report.has_errors
        
    

        
