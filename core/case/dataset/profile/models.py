from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel,Field, field_validator
from typing import Optional, Dict, Tuple, Set
from core.case.dataset.schema import NumericRange
from core.case.dataset.common import  DataFeatureType





class DataProfile(BaseModel) :
    """
    Statistical summary of the data, used for drift detection and baseline comparisons.
    Contains per‑feature statistics and a timestamp.
    
    Attributes:
        feature_profiles: Mapping from feature name to its statistical profile.
        computed_at: When the profile was generated.
    """
    feature_profiles : Dict[str, DataFeatureProfile]
    computed_at : Optional[datetime] = Field(None)
    
    def get_feature_names (self) -> Set[str]:
        return set(self.feature_profiles.keys())


    @field_validator('feature_profiles')
    def check_feature_names_match_keys(cls, v: dict) -> dict:
        """Ensure each DataFeatureProfile's internal name matches its dictionary key."""
        for key, profile in v.items():
            if profile.name != key:
                raise ValueError(
                    f"Feature name mismatch: key '{key}' does not match "
                    f"DataFeatureProfile.name '{profile.name}'"
                )
        return v



class DataFeatureProfile (BaseModel) :
    """
    Profile of a single feature within the dataset.
        
    Attributes:
        name: Feature name (must match the key in DataProfile).
        missing_rate: Fraction of missing values (0‑1).
        feature_stats: Type‑specific statistics (numeric, categorical, etc.).
    """
    name : str = Field (min_length=1)
    missing_rate : float = Field(ge=0.0,le=1.0)
    feature_stats : BaseStats 
    @field_validator('missing_rate')
    @classmethod
    def missing_rate_stats_match(cls, v):
        if v is None:
            return v
        if v == 1.0:
            raise ValueError("What do you want to know? all the values are missing")
        return v
        



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




