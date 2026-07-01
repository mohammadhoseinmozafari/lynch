
from case.application.use_cases.capture_environment import CaptureEnvironmentSnapshot
from case.infrastructure.services.pip_lock_capture_service import PipLockCapturer
from case.application.interfaces.environment_snapshot_capturer import EnvironmentSnapshotCapturer

class Container:
    """
    Dependency injection container for the application.
    """
    
    def __init__(self):
        # Register capturers (infrastructure)
        self.pip_lock_capturer: EnvironmentSnapshotCapturer = PipLockCapturer(timeout_seconds=30)
        
        # Register use cases (application)
        self.capture_pip_lock_usecase = CaptureEnvironmentSnapshot(self.pip_lock_capturer)
        
        # Future conda support (just add another instance)
        # self.conda_lock_capturer: LockCapturer = CondaLockCapturer()
        # self.capture_conda_lock_usecase = CaptureLockUseCase(self.conda_lock_capturer)