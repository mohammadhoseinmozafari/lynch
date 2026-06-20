from __future__ import annotations
from typing import Dict, List


from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.extractor import SignalExtractor
from core.signal.signal import Signal
from core.signal.type import SignalType


class MissingRateSignalExtractor (SignalExtractor) :

    def __init__(self) -> None:
        super().__init__()
        self.supporting_type = [
                ObservationType.COLUMN_MISSINGNESS,
                ObservationType.ROWS_MISSINGNESS,
                ObservationType.MISSINGNESS_DISTRIBUTION
            ]


    def extract (self , observations : List[Observation]) -> List[Signal]:
        signals: List[Signal] = []
        for observation in observations :
                if observation.payload.get("missing_rate", 0.0) > 0.8:
                    signals.append(self.build_signal(observation))

        return signals

    def extract_missing_rates (self , observations : List[Observation]) -> Dict[str, List[Signal]]:
        thresholds = {
            "low" : 0.0 ,
            "moderate" : 0.3,
            "high" : 0.6 ,
            "critical": 0.9    
        }
        critical_missing_rates : List[Signal] = []
        high_missing_rates: List[Signal] = []
        moderate_missing_rates : List[Signal] = []
        low_missing_rates : List[Signal] = []
        
        for obs in observations :
            rate = obs.payload["missing_rate"]
            if  rate > thresholds["low"] and rate <= thresholds["moderate"]:
                 low_missing_rates.append(
                      Signal (
                           signal_type= SignalType.LOW_MISSING_RATE,
                           value = rate,
                           source_observation_ids= [obs.id],
                           extractor_id= self.id
                      )
                 )
            elif  rate > thresholds["moderate"] and rate <= thresholds["high"]:
                 moderate_missing_rates.append(
                      Signal (
                           signal_type= SignalType.MODERATE_MISSING_RATE,
                           value = rate,
                           source_observation_ids= [obs.id],
                           extractor_id= self.id
                      )
                 )
            elif  rate > thresholds["high"] and rate <= thresholds["critical"]:
                 high_missing_rates.append(
                      Signal (
                           signal_type= SignalType.HIGH_MISSING_RATE,
                           value = rate,
                           source_observation_ids= [obs.id],
                           extractor_id= self.id
                      )
                 )
            
            elif  rate > thresholds["critical"]:
                 critical_missing_rates.append(
                      Signal (
                           signal_type= SignalType.CRITICAL_MISSING_RATE,
                           value = rate,
                           source_observation_ids= [obs.id],
                           extractor_id= self.id
                      )
                 )
        return {
             "low" : low_missing_rates,
             "moderate" : moderate_missing_rates,
             "high" : high_missing_rates,
             "critical" : critical_missing_rates
        }
    
    def extract_information_retained (self , observations : List[Observation]) -> List[Signal] :
        signals   : List[Signal] = []
        for obs in observations :
            info = 1 - obs.payload["missing_rate"]
            signals.append(
                 Signal(
                  signal_type= SignalType.COLUMN_INFORMATION_RETAINED,
                  value = info,
                  source_observation_ids= [obs.id],
                  extractor_id= self.id

             )
            )
        return signals
    
    def extract_estimation_uncertainty(self , observations : List[Observation]) -> List[Signal] :
        "Small datasets → unreliable missing rate."
        signals : List[Signal]  = []

        for obs in observations :
             rate = obs.payload["missing_rate"]
             total_count  = obs.payload ["total_count"]

             uncertainty  = (rate * (1-rate)) / total_count 
             signals.append (
                  Signal (
                       signal_type  = SignalType.MISSINGNESS_ESTIMATION_UNCERTAINTY,
                       value= uncertainty,
                       source_observation_ids= [obs.id],
                       extractor_id= self.id
                  )
             )
        return signals
             
          




    