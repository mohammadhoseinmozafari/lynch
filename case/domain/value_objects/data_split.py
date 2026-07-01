from pydantic import BaseModel, Field
from typing import Optional
from case.domain.enums.data_split_name import DataSplitName
from case.domain.value_objects.dataset_artifact import DataArtifactPointer
from case.domain.value_objects.data_schema import DataSchema
from case.domain.value_objects.data_profile import DataProfile
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


