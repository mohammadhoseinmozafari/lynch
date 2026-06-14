
from abc import ABC, abstractmethod
from typing import List

from core.domain.entities.evidence import Evidence

class BaseEvidenceCollector(ABC):

    @abstractmethod
    def collect (self, context) -> List[Evidence]:
        raise NotImplementedError