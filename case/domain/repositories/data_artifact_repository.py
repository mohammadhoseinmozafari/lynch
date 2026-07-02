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
    def resolve(self, pointer: DataArtifactPointer) -> pd.DataFrame:
        """Return the dataframe referenced by ``pointer``."""

    @abstractmethod
    def exists(self, pointer: DataArtifactPointer) -> bool:
        """Return whether ``pointer`` can be resolved by this repository."""

    @abstractmethod
    def delete(self, pointer: DataArtifactPointer) -> None:
        """Delete the artifact referenced by ``pointer``."""
