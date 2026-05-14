from __future__ import annotations
from pydantic import BaseModel,Field, field_validator
from core.case.domain.value_objects.data_profile_stats import BaseStats


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
        