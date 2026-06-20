from abc import ABC , abstractmethod
from typing import List
import uuid

from core.evidence.evidence import Evidence
from core.evidence.evidence_type import EvidenceType
from core.observation.observation import Observation
from core.observation.type import ObservationType
class EvidenceCollector(ABC):

    id: str
    evidence_type : EvidenceType
    supporting_type : ObservationType
    
    def __init__(self, normalizer) -> None:

        super().__init__()
        self.id = f"{self.__class__.__name__}_{uuid.uuid4()}"
        self.normalizer = normalizer
        
    @abstractmethod
    def collect (self, observations : List[Observation]) -> List[Evidence]:
        raise NotImplementedError
    