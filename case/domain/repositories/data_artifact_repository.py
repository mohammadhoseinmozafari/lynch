"""Repository contract for storing and resolving dataframe artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from case.domain.value_objects.dataset_artifact import DataArtifactPointer


class DataArtifactRepository(ABC):
    """Persistence boundary for pandas dataframe artifacts."""

    @abstractmethod
    def store(self, dataframe: pd.DataFrame) -> DataArtifactPointer:
        """Store ``dataframe`` and return a pointer containing no data."""

    @abstractmethod
    def get(self, object_id: str) -> pd.DataFrame:
        """Return the dataframe identified by ``object_id``."""

    @abstractmethod
    def exists(self, object_id: str) -> bool:
        """Return whether ``object_id`` can be resolved by this repository."""

    @abstractmethod
    def delete(self, object_id: str) -> None:
        """Delete the artifact identified by ``object_id``."""
