from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Finding (BaseModel):
    """
    The atomic domain primitive. Every investigation subsystem emits Finding objects.

    """

    id : UUID = Field(default_factory=lambda : uuid4())
    type : FindingType
    severity : Severity
    confidence : float
    title : str = Field (max_length=150)
    description : str = Field (max_length= 255)
    evidence : List[Evidence] = Field(default_factory=list)
    model_id : Optional[str] = None
    dataset_id : Optional[str] = None
    segment : Optional[Dict]


