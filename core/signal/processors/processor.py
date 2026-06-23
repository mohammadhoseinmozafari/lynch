from __future__ import annotations
from typing import List, Optional, Protocol, TypeVar
from core.observation.type import ObservationType
from core.signal.signal import  SignalTable
from core.signal.processors.context import ExecutionContext



C = TypeVar("C", bound= ExecutionContext,
            contravariant=True)
S = TypeVar ("S", bound= SignalTable,
             covariant=True)


class SignalProcessor(Protocol[C, S]):


    id: str  
    supporting_type: List[ObservationType]
    
    enabled_by_default: bool = True

    
    def dependencies(self) -> List[str]:
        return []

    def run(self, context: C) -> Optional[S]:
        pass


