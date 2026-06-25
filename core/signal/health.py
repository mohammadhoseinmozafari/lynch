from core.signal.type import HealthSignalType

from .signal import Signal, SignalEvent
from core.signal.enums import SignalCategory
from core.signal.enums import SubjectType
from pydantic import Field

class HealthSignal(Signal):


    category: SignalCategory = SignalCategory.HEALTH
    
    signal_type : HealthSignalType
    
    subject_type: SubjectType

    subject_name: str

    health_score: float = Field(ge = 0.0 , le = 1.0)

    dimensions: dict[str, float] = Field(default_factory=dict)

    def to_event (self) -> SignalEvent :
        return SignalEvent(
            id = self.id,
            category= self.category.value,
            signal_type=self.signal_type.value,
            subject_type= self.subject_type.value,
            subject_name= self.subject_name,
            value= self.health_score,
            confidence= self.confidence,
            source_observation_ids= self.source_observation_ids,
            
            payload= {
                "dimensions": self.dimensions
            },
            extractor_id= self.extractor_id,
            created_at= self.created_at
        )