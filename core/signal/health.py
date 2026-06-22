from .base import Signal, SignalCategory
from core.signal.subject_type import SubjectType
from pydantic import Field

class HealthSignal(Signal):

    category: SignalCategory = SignalCategory.HEALTH

    subject_type: SubjectType

    subject_name: str

    health_score: float = Field(ge = 0.0 , le = 1.0)

    dimensions: dict[str, float] = {}