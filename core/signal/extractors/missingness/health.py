from typing import Any, Callable, List
from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.signal.base import Signal
from core.signal.enums import  SubjectType
from core.signal.extractors.base import HealthSignalExtractor
from core.signal.health import HealthSignal
from core.signal.type import HealthSignalType

class ColumnMissingnessHealthSignalExtractor(HealthSignalExtractor):

    def __init__(self , config ) -> None:
        super().__init__()

        self.supporting_type = [ObservationType.COLUMN_MISSINGNESS]
        self.subject_type = SubjectType.FEATURE
        self.config = config
        
    def extract(self, observations: List[Observation]) -> List[HealthSignal]:

        signals: List[HealthSignal] = []


        for observation in observations:
            for extractor in self.extractors:
                signals.append (extractor(observation))


            

        return signals


    def extract_feature_completeness (self, observation : Observation)  -> HealthSignal: 
        rate = observation.payload["missing_rate"]
        completeness = 1 - rate

        return HealthSignal(
            signal_type= HealthSignalType.FEATURE_COMPLETENESS,
            value = completeness,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= observation.payload["column_name"],
            health_score= completeness
        )

    def extract_feature_missingness (self, observation : Observation)  -> HealthSignal:
        rate = observation.payload["missing_rate"]
        
        return HealthSignal(
            signal_type= HealthSignalType.FEATURE_MISSINGNESS,
            value = rate,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= observation.payload["column_name"],
            health_score= 1- rate
        )
        
    def extract_feature_usability (self, observation: Observation)  -> HealthSignal:
        rate = observation.payload["missing_rate"]
        f_usability = max (
            0,
            1- rate / self.config.acceptable_rate
        )

        return HealthSignal(
            signal_type= HealthSignalType.FEATURE_USABILITY,
            value = f_usability,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= observation.payload["column_name"],
            health_score= f_usability
        )

    def extract_feature_observation_confidence (self, observation : Observation)  -> HealthSignal:
        import math
        rate = observation.payload ["missing_rate"]
        total_count = observation.payload["total_count"]

        se = math.sqrt ((rate * (1- rate) / total_count))
        confidence =max(0 , 1 - 2*se)
        return HealthSignal(
            signal_type= HealthSignalType.FEATURE_OBSERVATION_CONFIDENCE,
            value = confidence,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= observation.payload["column_name"],
            health_score= confidence
        )
    
    @property
    def extractors(self) -> List[Callable]:
        extractors : List[Callable] = [
            self.extract_feature_completeness,
            self.extract_feature_missingness,
            self.extract_feature_usability, 
            self.extract_feature_observation_confidence,
        ]
        return extractors
    
    
     



class RowsMissingnessHealthSignalExtractor(HealthSignalExtractor):

    def __init__(self , config ) -> None:
        super().__init__()

        self.supporting_type = [ObservationType.ROWS_MISSINGNESS]
        self.subject_type = SubjectType.ROW
        self.config = config
        
    def extract(self, observations: List[Observation]) -> List[HealthSignal]:

        signals: List[HealthSignal] = []

        

        for observation in observations:
            for extractor in extractors:
                signals.append (extractor(observation))
        
        return signals

    def extract_dataset_row_integrity (self, observation : Observation) : 
        a = self.config.full_missing_row_integrity_param
        b = self.config.high_missing_row_integrity_param
        full_missing_rows_rate = observation.payload['full_missing_rows_rate']
        high_missing_rows_rate = observation.payload['high_missing_rows_rate']
        
        integrity = 1 - (
             a* full_missing_rows_rate +
             b * high_missing_rows_rate
        )
        integrity = max(
            0.0,
            min(
                1.0,
                integrity
            )
        )
        return HealthSignal(
            signal_type= HealthSignalType.DATASET_ROW_INTEGRITY,
            value = integrity,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "dataset_rows",
            health_score= integrity
        )
        
    def extract_complete_row_failure_rate (self, observation : Observation) :
        full_missing_rows_rate = observation.payload['full_missing_rows_rate']

        return HealthSignal(
            signal_type= HealthSignalType.COMPLETE_ROW_FAILURE_RATE,
            value = full_missing_rows_rate,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "full_missing_dataset_rows",
            health_score= 1- full_missing_rows_rate
        )
        
            
    def extract_partial_row_failure_rate (self, observation: Observation) :
        high_missing_rows_rate = observation.payload['high_missing_rows_rate']
        return HealthSignal(
            signal_type= HealthSignalType.PARTIAL_ROW_FAILURE_RATE,
            value = high_missing_rows_rate,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "high_missing_dataset_rows",
            health_score= 1 - high_missing_rows_rate
        )
    
    @property
    def extractors(self) -> List[Callable]:
        extractors : List[Callable] = [
            self.extract_dataset_row_integrity,
            self.extract_complete_row_failure_rate,
            self.extract_partial_row_failure_rate, 
        ]
        return extractors



class DatasetMissingnessHealthSignalExtractor(HealthSignalExtractor):

    def __init__(self , config ) -> None:
        super().__init__()

        self.supporting_type = [
            ObservationType.COLUMNS_MISSINGNESS_DISTRIBUTION                        
        ]
        self.subject_type = SubjectType.DATASET

        self.config = config
        
    def extract(self, observations: List[Observation]) -> List[HealthSignal]:

        signals: List[HealthSignal] = []

        

        for observation in observations:
            for extractor in extractors:
                signals.append (extractor(observation))
        
        return signals



    def extract_dataset_completeness (self, observation : Observation) : 
        completenss = 1 - observation.payload["mean_missing_rate"]

        return HealthSignal(
            signal_type= HealthSignalType.DATASET_COMPLETENESS,
            value = completenss,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "dataset",
            health_score= completenss
        )
    
    def extract_dataset_missingness (self, observation : Observation) : 
        missingness =  observation.payload["mean_missing_rate"]

        return HealthSignal(

            signal_type= HealthSignalType.DATASET_MISSINGNESS,
            value = missingness,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "dataset",
            health_score= 1-missingness
        )
    

    def extract_worst_feature_health (self, observation : Observation):
        max_missing_rate = observation.payload["max_missing_rate"]
        worst = 1 - max_missing_rate

        return HealthSignal(
            
            signal_type= HealthSignalType.WORST_FEATURE_HEALTH,
            value = worst,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "dataset",
            health_score= worst
        )
    
    def extract_best_feature_health (self, observation  : Observation) :
        min_missing_rate = observation.payload["min_missing_rate"]
        best = 1 - min_missing_rate

        return HealthSignal(
            
            signal_type= HealthSignalType.BEST_FEATURE_HEALTH,
            value = best,
            confidence= observation.reliability,
            source_observation_ids=[observation.id],
            extractor_id=self.id,
            subject_type=self.subject_type,
            subject_name= "dataset",
            health_score= best
        )
    
    @property
    def extractors(self) -> List[Callable]:
        extractors : List[Callable] = [
            self.extract_dataset_completeness,
            self.extract_dataset_missingness,
            self.extract_worst_feature_health,
            self.extract_best_feature_health 
        ]
        return extractors


         
