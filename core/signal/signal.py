from __future__ import annotations
from typing import List
from ulid import ulid
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
    id : str  = Field (default_factory= lambda : str(ulid()))

    signal_type : SignalType

    source_observation_ids :  List[str]
    
    normalization_record_id : str = ''
    
    extractor_id : str
