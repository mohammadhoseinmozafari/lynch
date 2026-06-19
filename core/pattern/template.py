
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from core.domain.entities.evidence import Evidence


class PatternTemplateCategory (str, Enum):
    MISSINGNESS = 'missingness'

class PatternLevel (Enum):
    ATOMIC = 1
    COMPOSITE = 2
    SEMANTIC = 3




class PatternTemplate(BaseModel):
    
    id: UUID = Field(default_factory= lambda : uuid4())

    name: str = Field (min_length= 1 , max_length= 100)
    description: str = Field(min_length= 1 , max_length= 255)

    category: PatternTemplateCategory

    level: PatternLevel
    input_types: List[Evidence]
    # observation types or parent pattern types

    tags: Optional[List[str]] = Field (default_factory= list)

    metadata: Optional[Dict]  = Field(default_factory= dict)