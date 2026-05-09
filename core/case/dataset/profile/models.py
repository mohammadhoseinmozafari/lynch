from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel,Field
from typing import List, Optional, Dict, Tuple
from enum import Enum
from core.case.dataset.schema import NumericRange


class FType (Enum) :
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    #DATETIME = "datetime"
    ## text_length
    ## geospatial


class DataProfileSnapshot (BaseModel) :
    row_counts : Dict[str, int]
    feature_stats : List[FeatureProfile]
    computed_at : Optional[datetime] = None
    drift_baseline : Optional[bool] = False
    


class FeatureProfile (BaseModel) :
    name : str = Field (min_length=1)
    missing_rate : float = Field(ge=0.0,le=1.0)
    feature_stats : Optional[BaseStats] 

class BaseStats (BaseModel) :
    feature_type : FType

class NumericStats (BaseStats):
    feature_type : FType = FType('numeric')
    range : Optional[NumericRange] = None
    mean : Optional[float] = None
    std : Optional [float] = Field(ge=0.0, default=None)
    quartiles : Optional[Tuple[float, float, float]]

class CategoricalStats (BaseStats) :
    feature_type : FType = FType('categorical')
    cardinality : Optional[int] = Field(ge=1.0, default=None)
    top_values : Optional [Dict[str, float]]


