
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class DataSchemaSnapshot (BaseModel) :
    features : List[FeatureSchema] = Field(min_length=1)
    target : Optional[str] = Field(min_length=1, default=None) 
    metadata_columns : Optional[List[str]] = None
    task_type : Optional[TaskType] = None

    
class FeatureSchema (BaseModel) :
    name : str = Field(min_length=1, max_length=255, pattern='^[a-zA-Z_][a-zA-Z0-9_]*$')
    human_readable_name : str = Field(min_length=1)
    dtype : DataType 
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


class DataType (Enum) :
    FLOAT64 = 'float64'
    INT64 = 'int64'
    BOOL = 'bool'
    DATETIME64 = 'datetime64'
    TEXT = 'text'

class TaskType(Enum) :
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    RANKING = 'ranking'
    CLUSTERING = 'clustering'
    GENERATIVE = 'generative'