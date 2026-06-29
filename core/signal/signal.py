from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from ulid import ulid
from pydantic import BaseModel, Field
from core.signal.enums import SignalCategory
import pandas as pd




@dataclass
class SignalEvent:
    id: str
    category: str
    signal_type: str
    subject_type: str | None
    subject_name: str | None
    value: float
    confidence: float
    source_observation_ids: List[str]
    payload: Dict[str, Any]
    extractor_id: str
    created_at: datetime


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

    @staticmethod
    def _enum_to_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    def to_event(self) -> SignalEvent:
        """Normalize any signal subtype into a persistence-ready event."""
        return SignalEvent(
            id=self.id,
            category=self._enum_to_value(self.category),
            signal_type=self._enum_to_value(self.signal_type),
            subject_type=self._enum_to_value(getattr(self, "subject_type", None))
            if getattr(self, "subject_type", None) is not None
            else None,
            subject_name=getattr(self, "subject_name", None),
            value=self.value,
            confidence=self.confidence,
            source_observation_ids=self.source_observation_ids,
            payload={},
            extractor_id=self.extractor_id,
            created_at=self.created_at,
        )

@dataclass
class SignalTable :
    table : pd.DataFrame