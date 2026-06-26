
from typing import List

from core.observation.collector import ObservationCollector, ObservationCollectorMethod
from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.enums import SubjectType


class HighColumnMissingness(ObservationCollector):
    
    def __init__(self , config ) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.HIGH_COLUMN_MISSINGNESS
        self.subject_type = SubjectType.FEATURE
        self.method_name = ObservationCollectorMethod.NULL_RATE
        self.config = config

    
    def collect (self, context) -> List[Observation]:
        
        profile = context.profile
        
        observations : List[Observation] = []
        threshold = self.config.high_threshold
        for feature_profile in profile.values():
            if feature_profile.missing_rate > threshold: 
                missingness_observation = self.build_observation(
                    payload= feature_profile.to_dict(),
                    subject_name=feature_profile.column_name,
                    reliability= 1.0
                    )

                observations.append(missingness_observation)

        
        
        
        



        return observations
    

        
    

    
class ModerateColumnMissingness(ObservationCollector):
    
    def __init__(self, config) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.MODERATE_COLUMN_MISSINGNESS
        self.method_name = ObservationCollectorMethod.NULL_RATE
        self.config = config
   
    def collect (self, context) -> List[Observation]:
        
        profile = context.profile
        
        observations       : List[Observation] = []
        high_threshold     = self.config.high_threshold
        moderate_threshold = self.config.moderate_threshold
        for feature_profile in profile.values():
            if (feature_profile.missing_rate > moderate_threshold 
                and 
                feature_profile.missing_rate <= high_threshold): 
                missingness_observation = self.build_observation(
                    payload= feature_profile.to_dict(),
                    subject_name=feature_profile.column_name,
                    reliability= 1.0
                    )

                observations.append(missingness_observation)

        
        
        
        



        return observations



class LowColumnMissingness(ObservationCollector):
    
    def __init__(self, config) -> None:
        super().__init__() 
        
        self.observation_type = ObservationType.LOW_COLUMN_MISSINGNESS
        self.method_name = ObservationCollectorMethod.NULL_RATE
        self.config = config
   
    def collect (self, context) -> List[Observation]:
        
        profile = context.profile
        
        observations       : List[Observation] = []
        moderate_threshold     = self.config.moderate_threshold
        low_threshold = self.config.low_threshold
        for feature_profile in profile.values():
            if (feature_profile.missing_rate > low_threshold 
                and 
                feature_profile.missing_rate <= moderate_threshold): 
                missingness_observation = self.build_observation(
                    payload= feature_profile.to_dict(),
                    subject_name=feature_profile.column_name,
                    reliability= 1.0
                    )

                observations.append(missingness_observation)

        
        
        
        



        return observations