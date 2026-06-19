from __future__ import annotations
from datetime import datetime
from pydantic import Field, BaseModel
from typing import Dict, Any
from uuid import uuid4
from core.observation.type import ObservationType

class Observation(BaseModel):
    
    id : str = Field (default_factory= lambda : str(uuid4()))

    type : ObservationType
    payload : Dict[Any , Any] = Field(default_factory=dict)
    reliability : float = Field(ge= 0.0 , le = 1.0)

    collected_at : datetime = Field(default_factory= datetime.now)
    collector_id : str


