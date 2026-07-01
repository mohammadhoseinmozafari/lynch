from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DataArtifactPointer(BaseModel, ABC):
    """Abstract handle used to resolve the data represented by a binding."""

    record_count: Optional[int] = Field(default=None, ge=0)
    column_count: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def resolve(self) -> Any:
        """Return the referenced data artifact."""


class InMemoryDataArtifactPointer(DataArtifactPointer):
    """A process-local pointer that retains a user's pandas DataFrame by reference."""

    key: str = Field(default_factory=lambda: f"memory://{uuid4()}")
    dataframe: pd.DataFrame = Field(exclude=True, repr=False)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "InMemoryDataArtifactPointer":
        return cls(
            dataframe=dataframe,
            record_count=len(dataframe.index),
            column_count=len(dataframe.columns),
        )

    def resolve(self) -> pd.DataFrame:
        return self.dataframe
