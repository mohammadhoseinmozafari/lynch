

from typing import List, Optional
from uuid import UUID

from core.domain.entities.evidence import Evidence
from core.domain.enums.evidence_type import EvidenceType


class EvidenceStore:

    def __init__(self, registry : BaseEvidenceRegistry) -> None:
        self.registry = registry

    def add(self, evidence: Evidence) -> None:

        self.registry.register (evidence)

    def add_many(self, evidences: List[Evidence]) -> None:
        
        for evidence in evidences:
            self.registry.register(evidence)
    
    def remove (self, evidence_id : str) -> None:
        self.registry.delete (evidence_id)

    def get(self, evidence_type) -> Optional[Evidence]:
        
        evidence = self.registry.get(evidence_type)
        
        return evidence

    def get_all (self) -> List[Evidence]:

        evidences = self.registry.get_all()
        return evidences
    
    def by_type (
            self, 
            evidence_type : EvidenceType
            ) -> List[Evidence] :
        
        evidences = self.registry.by_type (evidence_type)
        return evidences
    
    def by_collector (
            self, 
            collector : str
    ) -> List[Evidence]:
        
        evidences = self.registry.get_by_collector (collector)
        return evidences
    

    def filter (self, predicate) -> List[Evidence]:

        evidences = self.registry.filter (
            predicate
        )
        return evidences
    
