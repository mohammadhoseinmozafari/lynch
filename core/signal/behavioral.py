from core.signal.signal import Signal
from core.signal.enums import SignalCategory, SubjectType
from datetime import datetime

from core.signal.type import BehavioralSignalType

class BehavioralSignal(Signal):

    category = SignalCategory.BEHAVIORAL

    signal_type : BehavioralSignalType

    subject_type : SubjectType
    subject_name : str

    trend: str

    window_start: datetime | None = None

    window_end: datetime | None = None

    rate_of_change: float | None = None