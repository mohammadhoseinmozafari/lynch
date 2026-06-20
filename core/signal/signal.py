from __future__ import annotations
from typing import List
from uuid import uuid4
from pydantic import BaseModel, Field

from core.signal.type import SignalType



class Signal(BaseModel) :
    """
    A derived signal extracted from one or more observations.
    Hypothesis-agnostic, but semantically meaningful.

    Example:
        High Missing Rate
        Predictive Missingness
        Distribution Shift Signal.
    """ 
    id : str  = Field (default_factory= lambda : str(uuid4))

    signal_type : SignalType

    source_observation_ids :  List[str]
    
    extractor_id : str
