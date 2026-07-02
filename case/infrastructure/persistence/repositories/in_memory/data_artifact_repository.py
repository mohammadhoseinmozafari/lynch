"""Process-local dataframe artifact repository."""

from __future__ import annotations

from threading import RLock

import pandas as pd

from case.domain.repositories.artifact_repository import DataArtifactRepository
from case.domain.value_objects.dataset_artifact import (
    DataArtifactPointer,
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
            self._artifacts[pointer.key] = dataframe
        return pointer

    def resolve(self, pointer: DataArtifactPointer) -> pd.DataFrame:
        key = self._key_for(pointer)
        with self._lock:
            try:
                return self._artifacts[key]
            except KeyError as error:
                raise DataArtifactNotFoundError(
                    f"No in-memory dataframe exists for pointer {key!r}"
                ) from error

    def exists(self, pointer: DataArtifactPointer) -> bool:
        key = self._key_for(pointer)
        with self._lock:
            return key in self._artifacts

    def delete(self, pointer: DataArtifactPointer) -> None:
        key = self._key_for(pointer)
        with self._lock:
            try:
                del self._artifacts[key]
            except KeyError as error:
                raise DataArtifactNotFoundError(
                    f"No in-memory dataframe exists for pointer {key!r}"
                ) from error

    def clear(self) -> None:
        """Remove every artifact owned by this repository instance."""
        with self._lock:
            self._artifacts.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._artifacts)

    @staticmethod
    def _key_for(pointer: DataArtifactPointer) -> str:
        if not isinstance(pointer, InMemoryDataArtifactPointer):
            raise UnsupportedDataArtifactPointerError(
                "InMemoryDataArtifactRepository requires an "
                "InMemoryDataArtifactPointer"
            )
        return pointer.key
