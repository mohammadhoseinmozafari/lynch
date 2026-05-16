
from abc import ABC, abstractmethod
from core.case.domain.value_objects.environment_snapshot import EnvironmentSnapshot


class EnvironmentSnapshotCapturer(ABC):
    """
    Interface for scanning Python environments.
    This is the contract that infrastructure services must implement.
    """
    
    @abstractmethod
    def capture(self) -> EnvironmentSnapshot:
        """
        Capture A snapshot from current environment or workspace.
        
        
        Returns:
            EnvironmentSnapshot with specification for type of the capture
        
        Raises:
            EnvironmentNotDetectedError: If no environment can be detected
            PipLockCaptureError: If pip lock capture fails
        """
        pass