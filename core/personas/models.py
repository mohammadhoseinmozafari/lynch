from __future__ import annotations  
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, List


class Persona(str, Enum):
    TABULAR= 'tabular'
    VISION= 'vision'
    NLP= 'nlp'
    GENERIC= 'generic'


@dataclass
class PersonaDefinition :
    persona: str # e.g vision
    display_name : str # e.g "Image Deep Learning Researcher"
    description : Optional[str] # e.g "For computer vision models such as ResNet, ViT"
    target_audience : Optional[str]

    model_detection : ModelDetection
    recommended_algorithms : AlgorithmRecommendations
    strategy_templates : List[StrategyTemplate]





@dataclass
class ModelDetection :
    frameworks : List[str]  # ['scikit-learn' , 'pytorch']
    model_types : List[str] # ['classification', 'regression']


@dataclass
class AlgorithmRecommendations :
    local : AlgorithmConfig
    global_ : Optional[AlgorithmConfig]
    counterfactuals : Optional[CounterfactualConfig]

@dataclass
class AlgorithmConfig : 
    primary : str
    parameters : dict
    fallback : Optional[str]
    fallback_parameters : Optional[dict]

@dataclass
class StrategyTemplate : 
    name : str
    type : Literal["REALTIME", "BATCH_SCHEDULED", "ON_DEMAND", "EVENT_DRIVEN"]
    trigger : dict
    algorithms : List[str]
    scope : Literal['LOCAL','GLOBAL', 'BOTH']


@dataclass 
class CounterfactualConfig :
    enabled: bool
    algorithm : str
    parameters: dict

@dataclass
class ValidationRules :
    require_feature_descriptions : bool
    require_expected_range : bool
    warn_on_raw_names : bool
    allow_raw_feature_names: bool
    min_features_with_descriptions : float 

@dataclass
class ComputeProfile : 
    preffered_device : Literal["cpu","gpu"]
    max_memory_mb : int
    