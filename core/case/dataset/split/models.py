from pydantic import BaseModel, Field
from enum import Enum
from core.case.artifact import DataArtifactPointer
from core.case.dataset.profile import DataProfile
from core.case.dataset.schema import DataSchema
from typing import Optional


class DataSplitName(str,Enum) :
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DataSplit(BaseModel):
    """
    """
    split_name : DataSplitName
    split_artifact : DataArtifactPointer
    data_schema : DataSchema
    data_profile : Optional[DataProfile] = Field(None)
    row_count : Optional[int] = Field(ge=1)
    split_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)

    @property
    def name(self) -> str:
        return self.split_name.value


