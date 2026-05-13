from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from core.case.dataset.common import TaskType
from core.case.dataset.split import DataSplit, DataSplitName
from typing import Dict, Optional, List


class DatasetBinding(BaseModel) :
    dataset_name : Optional[str] = Field(None)
    data_splits : Dict[DataSplitName, DataSplit] = Field(min_length=1)
    drift_baseline : Optional[DataSplitName] = Field(default=None)
    task_type : Optional[TaskType] = Field(default=None)



    def get_split(self, split_name : str) -> Optional[DataSplit]:
        enum_key = DataSplitName(split_name)
        return self.data_splits.get(enum_key)
    
    def get_split_names(self) -> List[str]:
        return [split_name.value for  split_name in self.data_splits.keys()]

    
    model_config = ConfigDict(arbitrary_types_allowed=True)
