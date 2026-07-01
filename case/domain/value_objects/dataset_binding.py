from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Optional, List
from case.domain.enums.data_split_name import DataSplitName
from case.domain.enums.task_type import TaskType
from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.data_schema import DataSchema
class DatasetBinding(BaseModel) :

    dataset_name : Optional[str] = Field(None)
    dataset_schema : DataSchema
    dataset_profile: DataProfile
    drift_baseline : Optional[DataSplitName] = Field(default=None)
    task_type : Optional[TaskType] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
