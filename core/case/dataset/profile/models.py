from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel,Field
from typing import List, Optional, Dict, Tuple
from core.case.artifact import DataArtifactPointer
from core.case.dataset.schema import NumericRange
from core.case.dataset import DataSplitName, DataFeatureType






class DataProfile(BaseModel) :
    """
    Statistical summary of the data, used for drift detection and baseline comparisons.
    Attributes:
        splits: Splits of the data with their corresponding feature statistics.
        computed_at: When the profile was generated.
        drift_baseline: the given Split (e.g 'train') becomes the refrence for future
            drift detection for this model family.
    """
    splits : Dict[DataSplitName ,DataSplitProfile]
    computed_at : Optional[datetime] = None
    drift_baseline : Optional[DataSplitName] = DataSplitName.TRAIN
    

class DataSplitProfile(BaseModel):
    """
    Profile of a single data split (e.g train set).
    
    Attributes:
        split_name: Name of the split (currently supports train , val and test)
        split_artifact: Pointer to the data split artifact.
        row_count: Number of rows of the split.
        feature_stats: List of feature profiles of the data split.
        split_fraction: Fraction of split (e.g 0.8 for train split)
        checksum: checksum
    """
    split_name : DataSplitName
    split_artifact : DataArtifactPointer
    row_count : int = Field(ge=1)
    feature_stats : List[DataFeatureProfile]
    split_fraction : Optional[float] = Field(None, ge= 0.0, le=1.0)
    checksum : Optional[str] = Field(None, min_length=64, max_length=64)




class DataFeatureProfile (BaseModel) :
    """
    Profile of a single feature within the dataset.
        
    Attributes:
        name: name of the feature.
        missing_rate: ratio of missing values of the given feature.
        feature_stats: statistics (e.g mean, std, ...) of the given feature.
    """
    name : str = Field (min_length=1)
    missing_rate : float = Field(ge=0.0,le=1.0)
    feature_stats : Optional[BaseStats] 



class BaseStats (BaseModel) :
    """
    Base Model for data feature statistics.

    Attributes:
        feature_type: type of the feature (numerical, categorical, datetime, ...)
    """
    feature_type : DataFeatureType

class NumericStats (BaseStats):
    """
    Statistics of a numerical data feature.

    Attributes:
        feature_type: Type of the feature which is numerical. Cannot be changed.
        range: Range of the numerical feature. e.g (0,100)
        mean: Mean of the numerical feature.
        std: Standard deviation of the numerical feature.
        quartiles : Quartiles (25, 50, 75) of the numerical feature.
    """
    feature_type : DataFeatureType = Field(default=DataFeatureType.NUMERIC, frozen=True)
    range : Optional[NumericRange] = None
    mean : Optional[float] = None
    std : Optional [float] = Field(ge=0.0, default=None)
    quartiles : Optional[Tuple[float, float, float]] = Field(default=None)

class CategoricalStats (BaseStats) :
    """
    Statistics of a categorical data feature.
    
    Attributes:
        feature_type: Type of the feature which is categorical. Cannot be changed.
        cardinality: Number of unique values of the categorical feature.
        top_values: Values with highest appearance frequency.
    """
    feature_type : DataFeatureType = Field(default=DataFeatureType.CATEGORICAL, frozen=True)
    cardinality : Optional[int] = Field(ge=1.0, default=None)
    top_values : Optional [Dict[str, float]]




