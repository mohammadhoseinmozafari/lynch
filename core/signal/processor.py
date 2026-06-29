from __future__ import annotations

from typing import List, Optional, Protocol, TypeVar

from core.observation.type import ObservationType
from core.signal.signal import SignalTable


P = TypeVar("P", contravariant=True)
S = TypeVar("S", bound=SignalTable, covariant=True)


class SignalProcessor(Protocol[P, S]):


    id: str  
    supporting_type: List[ObservationType]
    
    enabled_by_default: bool = True

    
    def dependencies(self) -> List[str]:
        return []

    def run(self, profile: P) -> Optional[S]:
        pass


