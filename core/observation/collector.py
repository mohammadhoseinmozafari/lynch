from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List
import uuid
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
        self.id = f"{self.__class__.__name__}_{uuid.uuid4()}"
        
    @abstractmethod
    def collect (self, context) -> List[Observation]:
        raise NotImplementedError
    