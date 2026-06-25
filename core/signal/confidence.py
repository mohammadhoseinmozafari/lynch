from core.signal.signal import Signal
from core.signal.enums import SignalCategory
from core.signal.signal import SignalEvent


class ConfidenceSignal(Signal):

    category = SignalCategory.CONFIDENCE

    sample_size: int

    uncertainty: float

    measurement_noise: float | None = None

    def to_event(self) -> SignalEvent:
        event = super().to_event()
        event.payload = {
            "sample_size": self.sample_size,
            "uncertainty": self.uncertainty,
            "measurement_noise": self.measurement_noise,
        }
        return event