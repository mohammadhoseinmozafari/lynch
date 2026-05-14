from __future__ import annotations
from pydantic import BaseModel,Field, model_validator
from typing import Dict, Tuple
from core.case.domain.enums.data_feature_type import DataFeatureType
from core.case.domain.value_objects.data_profile_range import NumericRange


class BaseStats (BaseModel) :
    """Base for feature statistics, tagged by type."""
    feature_type : DataFeatureType

class NumericStats (BaseStats):
    """
    Statistics for a numerical feature.

    Attributes:
        feature_type: Numerical
        range: Range of the numerical feature. e.g (0,100)
        mean: Mean of the numerical feature.
        std: Standard deviation of the numerical feature.
        quartiles : Quartiles (25, 50, 75) of the numerical feature.
    """
    feature_type : DataFeatureType = Field(default=DataFeatureType.NUMERIC, frozen=True)
    range : NumericRange 
    mean : float 
    std : float 
    quartiles : Tuple[float, float, float]



class CategoricalStats (BaseStats) :
    """
    Statistics of a categorical data feature.
    
    Attributes:
        feature_type: Categorical
        cardinality: Number of unique values of the categorical feature.
        top_values: Values with highest appearance frequency.
    """
    feature_type : DataFeatureType = Field(default=DataFeatureType.CATEGORICAL, frozen=True)
    cardinality : int = Field(ge=1.0)
    top_values : Dict[str, float]
