
from typing import List

from core.observation.collector import ObservationCollector, ObservationCollectorMethod
from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.profilers.missingness import RowsMissingRateProfile
from core.signal.enums import SubjectType


class HighRowsMissingness(ObservationCollector) :

    def __init__(self, config) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.HIGH_ROWS_MISSINGNESS
        self.subject_type = SubjectType.ROW
        self.method_name = ObservationCollectorMethod.NULL_RATE

        self.config =config
    def collect (self , context) -> List[Observation]:
        
        profile : RowsMissingRateProfile= context.profile

        observations : List[Observation] = []

        if profile.high_missing_rate_rows_rate > 0 :

            observation = self.build_observation(
                subject_name="rows_stats",
                payload= profile.to_dict(),
                reliability= 1.0,
                )

            observations.append(observation)
        
        return observations
    


class FullRowsMissingness(ObservationCollector) :

    def __init__(self, config) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.FULL_ROWS_MISSINGNESS
        self.subject_type = SubjectType.ROW
        self.method_name = ObservationCollectorMethod.NULL_RATE

        self.config =config
    def collect (self , context) -> List[Observation]:
        
        profile : RowsMissingRateProfile= context.profile

        observations : List[Observation] = []

        if profile.full_missing_rows_rate > 0 :

            observation = self.build_observation(
                subject_name="rows_stats",
                payload= profile.to_dict(),
                reliability= 1.0,
                )

            observations.append(observation)
        
        return observations
    