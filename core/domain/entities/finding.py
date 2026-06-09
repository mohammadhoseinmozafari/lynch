from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from core.domain.entities.evidence import Evidence
from core.domain.entities.recommendation import Recommendation
from core.domain.enums.finding_type import FindingType
from core.domain.enums.severity import Severity


class Finding (BaseModel):
    """
    The atomic domain primitive. Every investigation subsystem emits
    Finding objects.

    Attributes:
        id: Unique identifier for this finding.
        type: What kind of finding this is.
        severity: How urgent this finding is.
        confidence: How confident the system is in this finding (0.0-1.0).
        title: Short human-readable title.
        description: Longer explanation of the finding.
        evidence: List of Evidence objects supporting this finding.
        recommendations: List of actionable Recommendation objects.
        model_id: The model this finding relates to, if any.
        dataset_id: The dataset this finding relates to, if any.
        segment: Optional dict defining a subpopulation this finding applies to.
        created_at: When this finding was generated.
    """

    id : UUID = Field(default_factory=lambda : uuid4())
    type : FindingType 
    severity : Severity
    confidence : float = Field (ge= 0.0, le = 1.0)
    title : str = Field (min_length=1 ,max_length=150)
    description : str = Field (min_length=1, max_length= 255)

    evidence : List[Evidence] = Field(min_length= 1 ,default_factory=list) 
    recommendations : List[Recommendation] = Field(default_factory= list)

    model_id : Optional[str] = None
    dataset_id : Optional[str] = None

    segment : Optional[Dict] = None

    created_at : datetime = Field(default_factory= datetime.now)


