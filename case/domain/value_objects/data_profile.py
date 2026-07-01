from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel,Field, field_validator
from typing import Optional, Dict, Set
from case.domain.value_objects.data_feature_profile import DataFeatureProfile





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
