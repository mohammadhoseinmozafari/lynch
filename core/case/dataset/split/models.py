from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
from core.case.artifact import DataArtifactPointer 


class SplitDefinition(BaseModel):
    split_name : SplitName
    split_artifact : DataArtifactPointer
    row_count : int = Field(ge=1)
    split_fraction : Optional[float] = Field(None, ge= 0.0, le=1.0)
    checksum : Optional[str] = Field(None, min_length=64, max_length=64)



class SplitName(Enum) :
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'