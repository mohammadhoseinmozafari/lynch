# src/infrastructure/services/pip_lock_capture_service.py
import subprocess
import sys

from core.case.application.interfaces.environment_snapshot_capturer import EnvironmentSnapshotCapturer
from core.case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from core.case.domain.enums.env_snapshot_type import EnvSnapshotType


class PipLockCapturer(EnvironmentSnapshotCapturer):
    """
    Captures pip lock from current Python environment.
    """
    def __init__(self, timeout_seconds :int = 30) -> None:
        
        self.timeout_seconds = timeout_seconds
        self._python_executable = sys.executable
    
    def capture(self) -> EnvironmentSnapshot:
        """
        Capture pip freeze output from current environment.
        
        Returns:
            EnvironmentSnapshot with pip freeze specification
        """
        try:
            # Run pip freeze
            result = subprocess.run(
                [self._python_executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_seconds
            )
            
            specification = result.stdout.strip()
            
            # Handle empty environment
            if not specification:
                specification = "# No packages installed"
            
            # Get Python version
            python_version = self._get_python_version()
            
            # Create snapshot (system_packages can be empty for MVP)
            return EnvironmentSnapshot(
                snapshot_type=EnvSnapshotType.PIP_LOCK,
                specification=specification,
                python_version=python_version,
                system_packages=[]  # MVP: empty list
            )
            
        except subprocess.CalledProcessError as e:
            raise ValueError(f"pip freeze failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise ValueError("pip freeze timed out")
        except Exception as e:
            raise ValueError(f"Failed to capture environment: {str(e)}")
        

    def _get_python_version(self) -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"