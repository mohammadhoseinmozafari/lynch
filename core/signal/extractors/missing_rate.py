from __future__ import annotations
from typing import List


from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.extractor import SignalExtractor
from core.signal.signal import Signal
from core.signal.type import SignalType


class HighMissingRateSignalExtractor (SignalExtractor) :

    def __init__(self) -> None:
        super().__init__()
        self.signal_type = SignalType.HIGH_MISSING_RATE
        self.supporting_type = ObservationType.COLUMN_MISSINGNESS


    def extract (self , observations : List[Observation]) -> List[Signal]:
        signals: List[Signal] = []
        for observation in observations :
                if observation.payload.get("missing_rate", 0.0) > 0.8:
                    signals.append(self.build_signal(observation))

        return signals
    

    def build_signal (self, observation : Observation) -> Signal:
          
         return Signal(
            signal_type= self.signal_type,
            source_observation_ids= [observation.id],
            extractor_id=self.id
              
         )
    

    


    