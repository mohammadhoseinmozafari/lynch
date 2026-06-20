from __future__ import annotations
from uuid import uuid4
from pydantic import BaseModel, Field

from core.evidence.evidence_type import EvidenceType



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
    id : str  = Field (default_factory= lambda : str(uuid4))

    evidence_type : EvidenceType
    source_observation_id :  str

    normalized_vector_id : str
    normalization_record_id : str
    
    collector_id : str

