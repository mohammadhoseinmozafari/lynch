from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List
from ulid import ulid
from pydantic import BaseModel, Field
from core.signal.enums import SignalCategory, SubjectType
from core.signal.type import SignalType
import pandas as pd




class Signal(BaseModel, ABC) :
    """
    A derived signal extracted from one or more observations.
    Hypothesis-agnostic, but semantically meaningful.

    Example:
        High Missing Rate
        Predictive Missingness
        Distribution Shift Signal.
    """ 
    id : str  = Field (default_factory= lambda : str(ulid()))

    category : SignalCategory

    signal_type : Any

    value : float = Field (ge = 0.0 , le = 1.0)

    confidence : float = Field (ge = 0.0 , le = 1.0)
    
    source_observation_ids :  List[str]
    
    normalization_record_id : str = ''
    
    extractor_id : str
    created_at : datetime = Field(default_factory= datetime.now)

@dataclass
class SignalTable :
    table : pd.DataFrame