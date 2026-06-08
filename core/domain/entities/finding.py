from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from core.domain.entities.evidence import Evidence
from core.domain.entities.recommendation import Recommendation
from core.domain.enums.finding_type import FindingType
from core.domain.enums.severity import Severity


class Finding (BaseModel):
    """
    The atomic domain primitive. Every investigation subsystem emits Finding objects.

    """

    id : UUID = Field(default_factory=lambda : uuid4())
    type : FindingType
    severity : Severity
    confidence : float = Field (ge= 0.0, le = 1.0)
    title : str = Field (max_length=150)
    description : str = Field (max_length= 255)
    evidence : List[Evidence] = Field(default_factory=list)
    recommendations : List[Recommendation]
    model_id : Optional[str] = None
    dataset_id : Optional[str] = None
    segment : Optional[Dict]


