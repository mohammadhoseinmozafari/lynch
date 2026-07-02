
import pytest
from pathlib import Path
import json
from case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from case.domain.enums.env_snapshot_type import EnvSnapshotType
from case.application.validators.environment_snapshot_validator import EnvironmentValidator
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
    
    def test_valid_system_packages_only(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'valid' / 'valid_system_packages_only.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                assert not report.has_errors, f"Expected no errors but got: {[e.code for e in report.errors]}"
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                for expected_code in expected_warning_codes:
                     assert expected_code in warning_codes

    def test_error_unpinned_python_version(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'error' / 'error_unpinned_python_version.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_invalid_python_version(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'error' / 'invalid_python_version.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes

    def test_missing_version_pin(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'error' / 'error_missing_version_pin_pip.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_unexact_version_pin(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'error' / 'error_unexact_version_pin_pip.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_invalid_pip_requirement(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'error' / 'error_invalid_pip_requirement.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_warning_dev_python_version(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_dev_python_version.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    
    def test_warning_local_python_version(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_local_python_version.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_warning_prerelease_python(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_prerelease_python.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_warning_deprecated_python(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_deprecated_python.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    
    def test_warning_extras_in_lock(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_extras_in_lock.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes

    def test_warning_markers_in_lock(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_markers_in_lock.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes

    def test_warning_url_dependency(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_url_dependency.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    
    def test_warning_missing_hashes_pip(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_missing_hashes_pip.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    def test_warning_unpinned_system_package(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_unpinned_system_package.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    
        
    def test_warning_version_range_system(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_version_range_system.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
    
    def test_warning_duplicate_dependencies(self, validator: EnvironmentValidator, validation_report: ValidationReport, load_test_case: Callable[...,Any])  -> None:
                test_dir =  Path(__file__).parent / 'fixtures' / 'warning' / 'warning_duplicate_dependencies.json'
                test_case= load_test_case(test_dir)
                snapshot= EnvironmentSnapshot.model_validate(test_case['input'])
                report = validator.validate(snapshot, validation_report)
                
                expected_error_codes = test_case['expected_errors']
                expected_warning_codes = test_case['expected_warnings']
                warning_codes = [warning.code for warning in report.warnings]
                error_codes = [error.code for error in report.errors]
                for expected_error_code in expected_error_codes:
                     assert expected_error_code in error_codes
                
                for expected_warn_code in expected_warning_codes:
                     assert expected_warn_code in warning_codes
        
    

        
