import json
import time
from typing import Any, List
from core.observation.observation import Observation
from core.observation.type import ObservationType
from storage.interfaces.observation_store import ObservationStore
from storage.sqlite.connection import SQLiteDB
class ObservationRepository(ObservationStore):
    def __init__(self, db : SQLiteDB) -> None:
        self.db = db

    def insert(self, observation : Observation)-> None: 
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO observations
                (id,  observation_type,  payload,  reliability, collected_at, collector_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                        observation.id,
                        observation.type,
                        json.dumps(observation.payload),
                        observation.reliability,
                        observation.collected_at,
                        observation.collector_id
                        
                        
                    ),
            )

    def insert_many(self, observations: List[Observation]) -> None:
        
        with self.db.connect() as conn:
        
            conn.executemany(
                """
                INSERT INTO observations
                (id,  observation_type,  payload,  reliability, collected_at, collector_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        observation.id,
                        observation.type,
                        json.dumps(observation.payload),
                        observation.reliability,
                        observation.collected_at,
                        observation.collector_id
                        
                        
                    )
                    for observation in observations
                ],
            )

    def fetch_by_type(self, observation_type: ObservationType) -> List[Any]:
        with self.db.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM observations
                WHERE observation_type = ?
                """,
                (observation_type,),
            ).fetchall()

    def fetch_all(self) -> List[Any]:
        with self.db.connect() as conn:
            return conn.execute(
                "SELECT * FROM observations"
            ).fetchall()