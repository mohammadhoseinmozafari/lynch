import json
import time


class ObservationRepository:
    def __init__(self, db):
        self.db = db

    def insert(self, observation):
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO observations
                (id, timestamp, observation_type, subject_type, subject_name, reliability, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(observation.id),
                    int(time.time()),
                    observation.type,
                    observation.subject_type,
                    observation.subject_name,
                    observation.reliability,
                    json.dumps(observation.payload),
                ),
            )

    def fetch_by_type(self, observation_type: str):
        with self.db.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM observations
                WHERE observation_type = ?
                """,
                (observation_type,),
            ).fetchall()

    def fetch_all(self):
        with self.db.connect() as conn:
            return conn.execute(
                "SELECT * FROM observations"
            ).fetchall()