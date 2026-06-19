from __future__ import annotations
from pydantic import Field, BaseModel
from typing import Dict
from uuid import uuid4
from core.observation.type import ObservationType

class Observation(BaseModel):
    
    id : str = Field (default_factory= lambda : str(uuid4()))

    type : ObservationType
    payload : Dict = Field(default_factory=dict)
    collector : str

