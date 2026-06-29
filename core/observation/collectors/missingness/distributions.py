from typing import List

from core.observation import (
    Observation,
    ObservationType,
    ObservationCollector, 
    ObservationCollectorMethod
    )
class RowsMissingnessDistribution(ObservationCollector) :

    def __init__(self , config) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.ROWS_MISSINGNESS_DISTRIBUTION
        self.method_name = ObservationCollectorMethod.NULL_RATE
        self.config = config
    
    def collect (self, context) -> List[Observation]:
        profile = context.profile
        observations : List[Observation] = []

        distribution_evidence = self.build_observation (profile)

        observations.append(distribution_evidence)
        
        return observations
    
    def build_observation (self , profile : MissingnessDistributionProfile ) -> Observation:
        

        return Observation (
            type = self.observation_type,
            payload = profile.to_dict(),
            reliability= 1.0,
            collector_id= self.id
            )

class ColumnsMissingnessDistributionObservationCollector (ObservationCollector) :

    def __init__(self, profiler: MissingRateProfiler) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.COLUMNS_MISSINGNESS_DISTRIBUTION
        self.method_name = ObservationCollectorMethod.NULL_RATE

        self._profiler = profiler
    
    def collect (self, context) -> List[Observation]:
        df = context.dataset

        observations : List[Observation] = []

        missingness_distribution_profile = self._profiler.profile_columns_distribution(df)

        distribution_evidence = self.build_observation (missingness_distribution_profile)

        observations.append(distribution_evidence)
        
        return observations
    
    def build_observation (self , profile : MissingnessDistributionProfile ) -> Observation:
        

        return Observation (
            type = self.observation_type,
            payload = profile.to_dict(),
            reliability= 1.0,
            collector_id= self.id
            )



