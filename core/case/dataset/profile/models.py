from __future__ import annotations
from pydantic import BaseModel,Field
from typing import List, Optional, Dict
from enum import Enum
from schema import NumericRange


class FType (Enum) :
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    #DATETIME = "datetime"
    ## text_length
    ## geospatial


class DataProfileSnapshot (BaseModel) :
    features : List[FeatureProfile]


class FeatureProfile (BaseModel) :
    name : str = Field (min_length=1)
    missing_rate : float = Field(ge=0.0,le=1.0)
    feature_stats : Optional[BaseStats] 

class BaseStats (BaseModel) :
    feature_type : FType

class NumericStats (BaseStats):
    feature_type : FType = FType('numeric')
    range : Optional[NumericRange] = None
    mean = Optional[float] = None
    std = Optional [float] = None

class CategoricalStats (BaseStats) :
    feature_type : FType = FType('categorical')
    cardinality : Optional[int] = None
    top_values : Optional [Dict[str, float]]


