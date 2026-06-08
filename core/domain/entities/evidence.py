
from typing import Dict, List, Optional

from pydantic import BaseModel

from core.domain.enums.evidence_type import EvidenceType


class Evidence(BaseModel) :
    type : EvidenceType
    payload: Dict
    samples : Optional[List[Dict]]
    statistics : Optional[Dict]
    chart_data: Optional[Dict]
