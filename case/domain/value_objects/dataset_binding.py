from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from case.domain.enums.data_split_name import DataSplitName
from case.domain.enums.task_type import TaskType
from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.data_schema import DataSchema
from case.domain.value_objects.dataset_artifact import DataArtifactPointer


class DatasetBinding(BaseModel):
    """Bind dataset metadata to a repository-resolved artifact pointer.

    The binding does not own or resolve dataframe data. Consumers pass
    ``artifact_pointer`` to the configured ``DataArtifactRepository``.
    """

    dataset_name: Optional[str] = Field(None)
    artifact_pointer: SerializeAsAny[DataArtifactPointer]
    dataset_schema: DataSchema
    dataset_profile: DataProfile
    drift_baseline: Optional[DataSplitName] = Field(default=None)
    task_type: Optional[TaskType] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
