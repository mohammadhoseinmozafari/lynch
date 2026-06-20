from __future__ import annotations

from abc import  ABC, abstractmethod

from typing import List
from ulid import ulid

from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.signal import Signal
from core.signal.type import SignalType

class SignalExtractor(ABC):


    id: str 
    signal_type : SignalType
    supporting_type : ObservationType
    
    def __init__(self) -> None:

        super().__init__()
        self.id = str(ulid())
        
    @abstractmethod
    def extract (self, observations : List[Observation]) -> List[Signal]:
        raise NotImplementedError
    