# src/infrastructure/persistence/repositories/postgres/training_run_repository.py
import psycopg2
from typing import Optional
from datetime import datetime
from src.domain.entities.training_run import TrainingRun
from src.domain.repositories.training_run_repository import TrainingRunRepository
from src.domain.value_objects.ids import VersionId
from ..postgres.helpers import serialize_uuid, deserialize_uuid, serialize_json, deserialize_json

class PostgresTrainingRunRepository(TrainingRunRepository):
    def __init__(self, conn: psycopg2.extensions.connection):
        self._conn = conn

    def insert(self, run: TrainingRun) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (
                    id, model_version_id, code_snapshot, environment_snapshot,
                    dataset_binding, hyperparameter_bundle, metric_bundle,
                    dataset_hash, training_run_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    serialize_uuid(run.id),
                    serialize_uuid(run.model_version_id),
                    serialize_json(run.code_snapshot),
                    serialize_json(run.environment_snapshot),
                    serialize_json(run.dataset_binding),
                    serialize_json(run.hyperparameter_bundle),
                    serialize_json(run.metric_bundle),
                    run.dataset_hash,
                    run.training_run_hash,
                    run.created_at.isoformat(),
                ),
            )

    def find_by_model_version_id(self, version_id: VersionId) -> Optional[TrainingRun]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM training_runs WHERE model_version_id = %s",
                (serialize_uuid(version_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_entity(row)

    def exists_by_hash(self, training_run_hash: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM training_runs WHERE training_run_hash = %s LIMIT 1",
                (training_run_hash,),
            )
            return cur.fetchone() is not None

    def _row_to_entity(self, row) -> TrainingRun:
        return TrainingRun(
            id=deserialize_uuid(row[0]),
            model_version_id=deserialize_uuid(row[1]),
            code_snapshot=deserialize_json(row[2]),
            environment_snapshot=deserialize_json(row[3]),
            dataset_binding=deserialize_json(row[4]),
            hyperparameter_bundle=deserialize_json(row[5]),
            metric_bundle=deserialize_json(row[6]),
            dataset_hash=row[7],
            training_run_hash=row[8],
            created_at=datetime.fromisoformat(row[9]),
        )