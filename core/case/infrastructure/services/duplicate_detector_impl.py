from core.case.application.interfaces.duplicate_detector import DuplicateDetector

class DuplicateDetectorImpl(DuplicateDetector):
    def __init__(self, repo: TrainingRunRepository) -> None:
        self.repo = repo

    def exists(self, training_run_hash: str) -> bool:
        return self.repo.exists_by_hash(training_run_hash)
        