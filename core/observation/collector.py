from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from core.observation.observation import Observation

class ObservationCollector(ABC):

    @abstractmethod
    def collect (self, context) -> List[Observation]:
        raise NotImplementedError