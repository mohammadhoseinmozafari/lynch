# src/infrastructure/persistence/repositories/postgres/artifact_repository.py
import psycopg2
from typing import Optional
from datetime import datetime
from src.domain.entities.model_artifact import ModelArtifact
from src.domain.repositories.artifact_repository import ArtifactRepository
from src.domain.value_objects.ids import VersionId
from ..postgres.helpers import serialize_uuid, deserialize_uuid, serialize_json, deserialize_json

class PostgresArtifactRepository(ArtifactRepository):
    def __init__(self, conn: psycopg2.extensions.connection):
        self._conn = conn

    def insert(self, artifact: ModelArtifact) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_artifacts (
                    id, model_version_id, storage_backend, key, checksum, size_bytes,
                    framework, artifact_type, manifest, tier, stored_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    serialize_uuid(artifact.id),
                    serialize_uuid(artifact.model_version_id),
                    artifact.storage_backend,
                    artifact.key,
                    artifact.checksum,
                    artifact.size_bytes,
                    artifact.framework,
                    artifact.artifact_type,
                    serialize_json(artifact.manifest),
                    artifact.tier,
                    artifact.stored_at.isoformat(),
                ),
            )

    def find_by_model_version_id(self, version_id: VersionId) -> Optional[ModelArtifact]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM model_artifacts WHERE model_version_id = %s",
                (serialize_uuid(version_id),),
            )
            row = cur.fetchone()
            return self._row_to_entity(row) if row else None

    def update_tier(self, artifact_id: str, new_tier: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE model_artifacts SET tier = %s WHERE id = %s",
                (new_tier, artifact_id),
            )

    def _row_to_entity(self, row) -> ModelArtifact:
        return ModelArtifact(
            id=deserialize_uuid(row[0]),
            model_version_id=deserialize_uuid(row[1]),
            storage_backend=row[2],
            key=row[3],
            checksum=row[4],
            size_bytes=row[5],
            framework=row[6],
            artifact_type=row[7],
            manifest=deserialize_json(row[8]),
            tier=row[9],
            stored_at=datetime.fromisoformat(row[10]),
        )