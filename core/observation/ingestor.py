

from typing import List
from core.observation.observation import Observation
from storage.interfaces.observation_store import ObservationStore


class ObservationIngestor:
    """
    This is the ONLY place collectors should talk to.
    """

    def __init__(self, store: ObservationStore):
        self.store = store

    def ingest(self, observations: List[Observation]):
        if not observations:
            return

        self.store.insert_many(observations)