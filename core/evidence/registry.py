from abc import ABC , abstractmethod
from typing import List

from core.domain.entities.evidence import Evidence
from core.domain.enums.evidence_type import EvidenceType

class BaseEvidenceRegistry (ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def register(self, evidence : Evidence) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get (self, evidence_id : str) -> Evidence :
        raise NotImplementedError
    
    @abstractmethod
    def get_all (self) -> List[Evidence]:
        raise NotImplementedError
    
    @abstractmethod
    def get_by_type(self, evidence_type: EvidenceType) -> List[Evidence]:
        raise NotImplementedError
    
    
    @abstractmethod
    def get_by_collector(self, evidence_collector: str) -> List[Evidence]:
        raise NotImplementedError
    
    @abstractmethod
    def filter(self, predicate) -> List[Evidence]:
        raise NotImplementedError
    
    @abstractmethod
    def delete (self, evidence_id : str) -> None:
        raise NotImplementedError
    
    