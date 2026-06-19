from threading import RLock
from typing import Callable, Dict, List, Set

from core.domain.entities.evidence import Evidence
from core.domain.enums.evidence_type import EvidenceType
from core.evidence.registry import BaseEvidenceRegistry


class InMemoryEvidenceRegistry(BaseEvidenceRegistry):

    def __init__(self) -> None:

        # Primary storage
        self._store: Dict[str, Evidence] = {}

        # Indexes (this is what makes it scalable in-memory)
        self._by_type: Dict[EvidenceType, Set[str]] 
        self._by_collector: Dict[str, Set[str]] 

        # NOTE: We should add subject index later if subject would be added to evidence class

        # Concurrency safety (SDK-level, not distributed)
        self._lock = RLock()
    
    def register(self, evidence: Evidence) -> None:

        with self._lock:

            if evidence.id in self._store:
                # You can also choose "update semantics"
                raise ValueError(f"Evidence already exists: {evidence.id}")

            self._store[evidence.id] = evidence

            self._by_type[evidence.type].add(evidence.id)

            self._by_collector[evidence.collector].add(evidence.id)
    
    def get(self, evidence_id: str) -> Evidence:

        with self._lock:

            if evidence_id not in self._store:
                raise KeyError(f"Evidence not found: {evidence_id}")

            return self._store[evidence_id]
        
    def get_all(self) -> List[Evidence]:

        with self._lock:

            return list(self._store.values())
        
    
    def get_by_type(self, evidence_type: EvidenceType) -> List[Evidence]:

        with self._lock:

            ids = self._by_type.get(evidence_type, set())

            return [self._store[i] for i in ids]
    
    
    def get_by_collector(self, evidence_collector: str) -> List[Evidence]:

        with self._lock:

            ids = self._by_collector.get(evidence_collector, set())

            return [self._store[i] for i in ids]
        
    def filter(self, predicate: Callable[[Evidence], bool]) -> List[Evidence]:

        with self._lock:

            return [
                ev for ev in self._store.values()
                if predicate(ev)
            ]
    
    def delete(self, evidence_id: str) -> None:

        with self._lock:

            if evidence_id not in self._store:
                return

            evidence = self._store.pop(evidence_id)

            self._by_type[evidence.type].discard(evidence_id)

            self._by_collector[evidence.collector].discard(evidence_id)

