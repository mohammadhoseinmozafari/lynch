from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from core.domain.enums.evidence_type import EvidenceType


class Evidence(BaseModel) :
    """
    Supporting data attached to a finding.

    Attributes:
        type: What kind of evidence this is (test result, sample rows, etc.).
        payload: Raw data from the investigation (test statistics, metrics).
        samples: Optional list of representative rows or instances.
        statistics: Optional summary statistics.
        chart_data: Optional pre-computed chart data for visualization.
    """
    type : EvidenceType 
    payload: Dict[str, Any] 
    samples : Optional[List[Dict]] = None
    statistics : Optional[Dict] = None
    chart_data: Optional[Dict] = None
