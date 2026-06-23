from core.signal.signal import Signal
from core.signal.enums import SignalCategory


class ConfidenceSignal(Signal):

    category = SignalCategory.CONFIDENCE

    sample_size: int

    uncertainty: float

    measurement_noise: float | None = None