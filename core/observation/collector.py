from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from core.observation.observation import Observation
from core.observation.type import ObservationType

class ObservationCollector(ABC):

    observation_type : ObservationType
    method_name : str

    @abstractmethod
    def collect (self, context) -> List[Observation]:
        raise NotImplementedError
    
    @abstractmethod
    def build_observation (self, Any) -> Observation:
        raise NotImplementedError