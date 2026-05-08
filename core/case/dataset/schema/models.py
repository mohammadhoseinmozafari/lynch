
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class DataSchemaSnapshot (BaseModel) :
    features : List[FeatureSchema]

class FeatureSchema (BaseModel) :
    name : str = Field(min_length=1)
    human_readable_name : str = Field(min_length=1)
    dtype : str ## should change to DType class
    expected_range : Optional[BaseRange] = None
    description : Optional[str] = Field(max_length=150, default="Feature description") 
    is_target : bool = False

class BaseRange (BaseModel) :
    feature_type : FType

class FType (str, Enum) :
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    #DATETIME = "datetime"
    ## text_length
    ## geospatial
class NumericRange (BaseRange):
    feature_type : FType = FType('numeric')
    min : Optional[float] = None
    max : Optional[float] = None

class CategoricalRange (BaseRange) :
    feature_type : FType =  FType('categorical')
    possible_values : List[str]