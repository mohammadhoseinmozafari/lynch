from __future__ import annotations
from typing import List, Optional, Protocol, TypeVar
from core.signal.base import Signal
from core.signal.processor.context import ExecutionContext



C = TypeVar("C", bound= ExecutionContext,
            contravariant=True)
S = TypeVar ("S", bound= Signal,
             covariant=True)


class SignalProcessor(Protocol[C, S]):


    id: str 
    
    name : str 
    
    enabled_by_default: bool = True

    
    def dependencies(self) -> List[str]:
        return []

    def run(self, ctx: C) -> Optional[S]:
        pass


