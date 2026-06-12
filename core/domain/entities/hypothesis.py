from dataclasses import dataclass, field
from typing import List, Set
from uuid import UUID, uuid4

from core.domain.entities.evidence import Evidence
from core.domain.enums.hypothesis_status import HypothesisStatus
from core.domain.enums.hypothesis_type import HypothesisType


@dataclass
class BeliefState : 
    prior : float
    posterior : float
    confidence : float

@dataclass
class Hypothesis :

    type : HypothesisType
    target_columns : List[str]

    belief : BeliefState

    supporting_evidence : Set[Evidence]
    opposing_evidence : Set[Evidence]

    status: HypothesisStatus

    reasoning_trace : List[str]
    
    id : UUID = field(default_factory= lambda : uuid4())


