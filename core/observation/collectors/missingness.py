from __future__ import annotations


from typing import List

import pandas as pd

from core.observation.collector import (
    ObservationCollector,
    ObservationCollectorMethod
)
from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.profilers.missingness.profiler import  MissingRateProfiler
from core.profilers.missingness.models import (
    ColumnMissingRateProfile,  
    MissingnessDistributionProfile, 
    MissingnessDistributionProfile, 
    RowsMissingRateProfile, 
    RowsMissingRateProfile
)








class ColumnMissingnessObservationCollector(ObservationCollector):
    
    def __init__(self, profiler: MissingRateProfiler) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.COLUMN_MISSINGNESS
        self.method_name = ObservationCollectorMethod.NULL_RATE

        self._profiler = profiler

    
    def collect (self, context) -> List[Observation]:
        
        df : pd.DataFrame = context.dataset
        
        observations : List[Observation] = []
        
        missingness_profile= self._profiler.profile_columns(df)

        for column, profile in missingness_profile.items():
            missingness_observation = self.build_observation(profile)

            observations.append(missingness_observation)

        
        
        
        



        return observations
    

        
    def build_observation (self, profile :  ColumnMissingRateProfile) -> Observation:

        return Observation(
            type = ObservationType.COLUMN_MISSINGNESS,
            payload = profile.to_dict(),
            reliability= 1.0, 
            collector_id= self.id
        )

    
    

class RowsMissingnessObservationCollector (ObservationCollector) :

    def __init__(self, profiler: MissingRateProfiler) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.ROWS_MISSINGNESS
        self.method_name = ObservationCollectorMethod.NULL_RATE

        self._profiler = profiler

    def collect (self , context) -> List[Observation]:
        
        df = context.dataset
        sample_size = context.sample_size

        observations : List[Observation] = []

        missingness_profile = self._profiler.profile_rows(df, sample_size)

        missingness_observation = self.build_observation(missingness_profile)

        observations.append(missingness_observation)
        
        return observations
    
    def build_observation (self , profile : RowsMissingRateProfile ) -> Observation:
        
        
        

        return Observation (
            type = self.observation_type,
            payload = profile.to_dict(),
            reliability= 1.0,
            collector_id= self.id

        )

class RowsMissingnessDistributionObservationCollector (ObservationCollector) :

    def __init__(self, profiler: MissingRateProfiler) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.ROWS_MISSINGNESS_DISTRIBUTION
        self.method_name = ObservationCollectorMethod.NULL_RATE

        self._profiler = profiler
    
    def collect (self, context) -> List[Observation]:
        df = context.dataset

        observations : List[Observation] = []

        missingness_distribution_profile = self._profiler.profile_rows_distribution(df)

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









