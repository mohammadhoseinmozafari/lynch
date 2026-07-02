from __future__ import annotations

from abc import ABC
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DataArtifactPointer(BaseModel, ABC):
    """Serializable identity and metadata for a stored dataset artifact.

    Pointers deliberately contain no artifact data and perform no I/O. A
    ``DataArtifactRepository`` is responsible for resolving them.
    """

    object_id: str = Field(min_length=1)
    record_count: Optional[int] = Field(default=None, ge=0)
    column_count: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class InMemoryDataArtifactPointer(DataArtifactPointer):
    """Identifier for a dataframe owned by an in-memory repository."""

    object_id: str = Field(
        default_factory=lambda: f"memory://{uuid4()}",
        pattern=r"^memory://.+$",
    )
