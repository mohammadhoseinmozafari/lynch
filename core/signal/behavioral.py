from core.signal.signal import Signal
from core.signal.enums import SignalCategory, SubjectType
from datetime import datetime

from core.signal.type import BehavioralSignalType
from core.signal.signal import SignalEvent

class BehavioralSignal(Signal):

    category = SignalCategory.BEHAVIORAL

    signal_type : BehavioralSignalType

    subject_type : SubjectType
    subject_name : str

    trend: str

    window_start: datetime | None = None

    window_end: datetime | None = None

    rate_of_change: float | None = None

    def to_event(self) -> SignalEvent:
        event = super().to_event()
        event.payload = {
            "trend": self.trend,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "rate_of_change": self.rate_of_change,
        }
        return event