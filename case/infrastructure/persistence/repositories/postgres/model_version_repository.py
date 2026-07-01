# core.case/infrastructure/persistence/repositories/postgres/model_version_repository.py
import psycopg2
from typing import Optional
from core.case.domain.entities.model_version import ModelVersion
from case.domain.repositories.model_version_repository import ModelVersionRepository
from core.case.domain.value_objects.ids import FamilyId, VersionId
from core.case.domain.value_objects.enums import ModelStatus, Completeness
from ..postgres.helpers import serialize_uuid, deserialize_uuid, serialize_datetime

class PostgresModelVersionRepository(ModelVersionRepository):
    def __init__(self, conn: psycopg2.extensions.connection):
        self._conn = conn

    def insert(self, version: ModelVersion) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_versions (
                    version_id, family_id, project_id, version_number, status,
                    persona, version_hash, parent_version_id, completeness,
                    interpretability_score, validation_warnings, created_at, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    serialize_uuid(version.version_id),
                    serialize_uuid(version.family_id),
                    serialize_uuid(version.project_id),
                    version.version_number,
                    version.status.value,
                    version.persona,
                    version.version_hash,
                    serialize_uuid(version.parent_version_id) if version.parent_version_id else None,
                    version.completeness.value,
                    version.interpretability_score,
                    ",".join(version.validation_warnings),
                    serialize_datetime(version.created_at),
                    version.created_by,
                ),
            )

    def find_by_id(self, version_id: VersionId) -> Optional[ModelVersion]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM model_versions WHERE version_id = %s",
                (serialize_uuid(version_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_entity(row)

    def find_by_family_and_version(self, family_id: FamilyId, version_number: int) -> Optional[ModelVersion]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM model_versions WHERE family_id = %s AND version_number = %s",
                (serialize_uuid(family_id), version_number),
            )
            row = cur.fetchone()
            return self._row_to_entity(row) if row else None

    def get_next_version_number(self, family_id: FamilyId) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM model_versions WHERE family_id = %s",
                (serialize_uuid(family_id),),
            )
            return cur.fetchone()[0]

    def _row_to_entity(self, row) -> ModelVersion:
        # row is a tuple; order matches SELECT * columns.
        return ModelVersion(
            version_id=deserialize_uuid(row[0]),
            family_id=deserialize_uuid(row[1]),
            project_id=deserialize_uuid(row[2]),
            version_number=row[3],
            status=ModelStatus(row[4]),
            persona=row[5],
            version_hash=row[6],
            parent_version_id=deserialize_uuid(row[7]) if row[7] else None,
            completeness=Completeness(row[8]),
            interpretability_score=row[9],
            validation_warnings=row[10].split(",") if row[10] else [],
            created_at=deserialize_datetime(row[11]),
            created_by=row[12],
        )