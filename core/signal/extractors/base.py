from __future__ import annotations

from abc import  ABC, abstractmethod

from typing import Any, List
from ulid import ulid

from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.base import Signal
from core.signal.enums import SignalCategory
from core.signal.health import HealthSignal
from core.signal.type import SignalType

class SignalExtractor(ABC):


    id: str 
    supporting_type : List[ObservationType]
    
    def __init__(self) -> None:

        super().__init__()
        self.id = str(ulid())
        
    @abstractmethod
    def extract (self, observations : List[Observation]) -> List:
        raise NotImplementedError
    
    @property
    def extractors(self) -> Any:
        raise NotImplementedError

class HealthSignalExtractor (SignalExtractor):
    
    def __init__(self) -> None:
        super().__init__()
        self.signal_category = SignalCategory.HEALTH
    
    @abstractmethod
    def extract(self, observations: List[Observation]) -> List[HealthSignal]:
        raise NotImplementedError

    @property
    def extractors(self) -> Any:
        return super().extractors

