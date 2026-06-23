from core.signal.type import HealthSignalType

from .signal import Signal, SignalCategory
from core.signal.enums import SubjectType
from pydantic import Field

class HealthSignal(Signal):


    category: SignalCategory = SignalCategory.HEALTH
    
    signal_type : HealthSignalType
    
    subject_type: SubjectType

    subject_name: str

    health_score: float = Field(ge = 0.0 , le = 1.0)

    dimensions: dict[str, float] = {}