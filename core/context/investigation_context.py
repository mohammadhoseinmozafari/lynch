"""Execution context shared by dataset investigation profilers."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.dataset_binding import DatasetBinding


class InvestigationContext(BaseModel):
    """Provide profilers with data and run-scoped investigation state.

    Artifact resolution is centralized here so profilers depend only on this
    context abstraction. The context does not know a concrete repository
    implementation; it requires only ``store(dataframe)`` for construction and
    ``get(object_id)`` for resolution.
    """

    dataset_binding: DatasetBinding | None = None
    profile: DataProfile
    artifact_store: Any
    artifact_pointer: Any
    runtime_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        *,
        profile: DataProfile,
        artifact_store: Any,
        dataset_binding: Optional[DatasetBinding] = None,
        runtime_params: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "InvestigationContext":
        """Store a raw dataframe and return a ready investigation context."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        store = getattr(artifact_store, "store", None)
        if not callable(store):
            raise TypeError("artifact_store must provide store(dataframe)")

        pointer = store(dataframe)
        return cls(
            dataset_binding=dataset_binding,
            profile=profile,
            artifact_store=artifact_store,
            artifact_pointer=pointer,
            runtime_params=dict(runtime_params or {}),
            metadata=dict(metadata or {}),
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Resolve and return the context's pandas dataframe artifact."""
        object_id = getattr(self.artifact_pointer, "object_id", None)
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("artifact_pointer must provide a non-empty object_id")

        get_artifact = getattr(self.artifact_store, "get", None)
        if not callable(get_artifact):
            raise TypeError("artifact_store must provide get(object_id)")

        dataframe = get_artifact(object_id)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("resolved artifact must be a pandas DataFrame")
        return dataframe

    def get_param(self, key: str, default: Any = None) -> Any:
        """Read a run-scoped parameter without mutating the context."""
        return self.runtime_params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """Set a run-scoped parameter for subsequent investigation steps."""
        self.runtime_params[key] = value
