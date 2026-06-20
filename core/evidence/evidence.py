from __future__ import annotations
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from core.domain.enums.evidence_type import EvidenceType
from core.observation.observation import Observation


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
    id : str #same as the source observation id
    source_observation_id :  str
    normalized_vector_id : str
    normalization_record_id : str
    promoted_by : str

