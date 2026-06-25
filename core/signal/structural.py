from core.signal.signal import Signal
from core.signal.enums import SignalCategory
from core.signal.signal import SignalEvent


class StructuralSignal(Signal):

    category = SignalCategory.STRUCTURAL

    scope: str

    affected_subjects: list[str]

    pattern_strength: float

    def to_event(self) -> SignalEvent:
        event = super().to_event()
        event.payload = {
            "scope": self.scope,
            "affected_subjects": self.affected_subjects,
            "pattern_strength": self.pattern_strength,
        }
        return event