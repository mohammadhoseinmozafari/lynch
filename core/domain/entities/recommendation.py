from typing import Optional
from pydantic import BaseModel, Field

from core.domain.enums.effort_level import EffortLevel


class Recommendation (BaseModel) :
    action : str  = Field (max_length=150)
    rational : str = Field(max_length=255)
    expected_impact : str = Field(max_length= 255)
    efforte : EffortLevel
    priority : int = Field(ge= 1)
    estimated_metric_gain : Optional[float] = Field(ge = 0.0, default= None)
