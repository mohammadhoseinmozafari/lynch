"""Process-local dataframe artifact repository."""

from __future__ import annotations

from threading import RLock

import pandas as pd

from case.domain.repositories.data_artifact_repository import DataArtifactRepository
from case.domain.value_objects.dataset_artifact import (
    InMemoryDataArtifactPointer,
)


class DataArtifactNotFoundError(KeyError):
    """Raised when an artifact pointer is unknown to a repository."""


class UnsupportedDataArtifactPointerError(TypeError):
    """Raised when a repository receives an incompatible pointer type."""


class InMemoryDataArtifactRepository(DataArtifactRepository):
    """Store pandas dataframes by reference for the lifetime of this instance.

    The repository, rather than the pointer, owns the strong dataframe
    reference. Operations are protected by a re-entrant lock so independent
    threads cannot observe partially completed store/delete operations.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, pd.DataFrame] = {}
        self._lock = RLock()

    def store(self, dataframe: pd.DataFrame) -> InMemoryDataArtifactPointer:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("artifact must be a pandas DataFrame")

        pointer = InMemoryDataArtifactPointer(
            record_count=len(dataframe.index),
            column_count=len(dataframe.columns),
        )
        with self._lock:
            self._artifacts[pointer.object_id] = dataframe
        return pointer

    def get(self, object_id: str) -> pd.DataFrame:
        object_id = self._validate_object_id(object_id)
        with self._lock:
            try:
                return self._artifacts[object_id]
            except KeyError as error:
                raise DataArtifactNotFoundError(
                    f"No in-memory dataframe exists for object {object_id!r}"
                ) from error

    def exists(self, object_id: str) -> bool:
        object_id = self._validate_object_id(object_id)
        with self._lock:
            return object_id in self._artifacts

    def delete(self, object_id: str) -> None:
        object_id = self._validate_object_id(object_id)
        with self._lock:
            try:
                del self._artifacts[object_id]
            except KeyError as error:
                raise DataArtifactNotFoundError(
                    f"No in-memory dataframe exists for object {object_id!r}"
                ) from error

    def clear(self) -> None:
        """Remove every artifact owned by this repository instance."""
        with self._lock:
            self._artifacts.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._artifacts)

    @staticmethod
    def _validate_object_id(object_id: str) -> str:
        if not isinstance(object_id, str) or not object_id.startswith("memory://"):
            raise UnsupportedDataArtifactPointerError(
                "InMemoryDataArtifactRepository requires a memory:// object_id"
            )
        return object_id
