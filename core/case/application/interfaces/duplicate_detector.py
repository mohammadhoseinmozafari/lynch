from abc import ABC, abstractmethod

class DuplicateDetector(ABC):
    @abstractmethod
    def exists(self, training_run_hash: str) -> bool:
        raise NotImplementedError()