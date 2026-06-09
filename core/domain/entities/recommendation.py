from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from core.domain.enums.effort_level import EffortLevel


class Recommendation (BaseModel) :
    """
    An actionable suggestion attached to a finding.

    Attributes:
        action: What the user should do.
        rationale: Why this action is recommended.
        expected_impact: What improvement to expect (qualitative).
        effort: How much work is required (LOW, MEDIUM, HIGH).
        priority: Lower number = higher priority (1 is most urgent).
        estimated_metric_gain: Optional quantitative estimate of improvement.
    """
    action : str  = Field (min_length=1 , max_length=150)
    rational : str = Field(min_length=1 ,max_length=255)
    expected_impact : str = Field(min_length= 1 ,max_length= 255)
    effort : EffortLevel 
    priority : int = Field(ge= 1)
    estimated_metric_gain : Optional[float] = None