from core.signal.base import Signal
from core.signal.enums import SignalCategory
from datetime import datetime

class BehavioralSignal(Signal):

    category = SignalCategory.BEHAVIORAL

    trend: str

    window_start: datetime | None = None

    window_end: datetime | None = None

    rate_of_change: float | None = None