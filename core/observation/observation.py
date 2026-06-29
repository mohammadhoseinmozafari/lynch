from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pydantic import Field, BaseModel
from typing import Dict, Any
from ulid import ulid
from core.observation.type import ObservationType
import polars as pl

from core.signal.enums import SubjectType
class Observation(BaseModel):
    
    id : str = Field (default_factory= lambda : str(ulid()))

    type : ObservationType
    
    subject_type : SubjectType
    subject_name : str
    
    payload : Dict[Any , Any] = Field(default_factory=dict)
    reliability : float = Field(ge= 0.0 , le = 1.0)

    collected_at : datetime = Field(default_factory= datetime.now)
    collector_id : str


@dataclass
class ObservationBatch :
    df : pl.DataFrame 