from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID , uuid4
from pydantic import BaseModel, Field



class PatternInstance(BaseModel):

    id: UUID = Field (default_factory= lambda: uuid4())

    template_id: UUID = Field (default_factory= lambda: uuid4())

    subject: str # NOTE: should be implemented later


    confidence: float

    supporting_evidences: list[str] # NOTE: should be implemented later

    supporting_patterns: list[str] # NOTE: should be implemented later

    created_at : datetime = Field(default_factory= datetime.now)

    metadata : Optional[Dict]  = Field(default_factory= dict)