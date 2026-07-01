
from case.application.interfaces.environment_snapshot_capturer import EnvironmentSnapshotCapturer
from case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
class CaptureEnvironmentSnapshot:
    def __init__(self, capturer: EnvironmentSnapshotCapturer ) -> None:
        self.capturer= capturer
    
    def execute(self) -> EnvironmentSnapshot:
        try:
            return self.capturer.capture()
        except Exception as e:
            raise ValueError(e)