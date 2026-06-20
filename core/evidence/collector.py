from abc import ABC , abstractmethod
from typing import List
import uuid

from core.evidence.evidence import Evidence
from core.evidence.evidence_type import EvidenceType
class EvidenceCollector(ABC):

    id: str
    evidence_type : EvidenceType
    
    def __init__(self) -> None:

        super().__init__()
        self.id = f"{self.__class__.__name__}_{uuid.uuid4()}"
        
    @abstractmethod
    def collect (self, context) -> List[Evidence]:
        raise NotImplementedError
    