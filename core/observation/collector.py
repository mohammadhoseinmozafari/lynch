from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from ulid import ulid
from core.observation.observation import Observation
from core.observation.type import ObservationType

class ObservationCollectorMethod(str, Enum):
    NULL_RATE = "null_rate"
class ObservationCollector(ABC):
    id: str
    observation_type : ObservationType
    method_name : ObservationCollectorMethod
    
    def __init__(self) -> None:

        super().__init__()
        self.id = str(ulid())
        
    @abstractmethod
    def collect (self, context) -> List[Observation]:
        raise NotImplementedError
    