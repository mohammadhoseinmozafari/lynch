from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List
from ulid import ulid
from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.enums import SubjectType

class ObservationCollectorMethod(str, Enum):
    NULL_RATE = "null_rate"
class ObservationCollector(ABC):
    id: str
    observation_type : ObservationType
    method_name : ObservationCollectorMethod
    subject_type : SubjectType
    
    def __init__(self) -> None:

        super().__init__()
        self.id = str(ulid())
        
    @abstractmethod
    def collect (self, context) -> List[Observation]:
        raise NotImplementedError
    
    def build_observation (self, subject_name: str, payload : Dict, reliability : float ):
        return Observation(
            id = self.id,
            type = self.observation_type,
            subject_type= self.subject_type,
            subject_name= subject_name,
            payload= payload,
            reliability= reliability,
            collector_id= self.id
        )