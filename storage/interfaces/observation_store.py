from typing import Any, List, Protocol

from core.observation.observation import Observation
from core.observation.type import ObservationType


class ObservationStore(Protocol):
    def insert_many (self, observations: List[Observation])-> None: 
        ...
    def insert(self, observation: Observation):
        ...
    def fetch_by_type(self, observation_type: ObservationType)-> List[Any]: 
        ...
