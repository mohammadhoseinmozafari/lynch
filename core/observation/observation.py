from pydantic import Field, BaseModel
from enum import Enum
from typing import Any, Dict
from uuid import uuid4

class ObservationType (str, Enum):
    COLUMN_MISSINGNESS = "COLUMN_MISSINGNESS" 
    ROWS_MISSINGNESS =  "ROWS_MISSINGNESS" 
    MISSINGNESS_DISTRIBUTION = "MISSINGNESS_DISTRIBUTION"


class Observation(BaseModel):
    
    id : str = Field (default_factory= lambda : str(uuid4()))

    type : ObservationType
    payload : Dict = Field(default_factory=dict)
    collector : str

