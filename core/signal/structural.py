from core.signal.base import Signal
from core.signal.enums import SignalCategory


class StructuralSignal(Signal):

    category = SignalCategory.STRUCTURAL

    scope: str

    affected_subjects: list[str]

    pattern_strength: float