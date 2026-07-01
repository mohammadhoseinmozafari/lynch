from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional, Set

from pydantic import BaseModel, Field, field_validator

from case.domain.value_objects.data_feature_profile import DataFeatureProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace





class DataProfile(BaseModel) :
    """
    Statistical summary of the data, used for drift detection and baseline comparisons.
    Contains per‑feature statistics and a timestamp.
    
    Attributes:
        feature_profiles: Mapping from feature name to its statistical profile.
        computed_at: When the profile was generated.
    """
    feature_profiles : Dict[str, DataFeatureProfile]
    namespaces : Dict[str, ProfileNamespace] = Field(default_factory=dict)
    capabilities : Set[str] = Field(default_factory=set)
    computed_at : Optional[datetime] = Field(default_factory= datetime.now)
    updated_at : Optional[datetime] = None

    def touch (self) -> None:
        self.updated_at = datetime.now()

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def add_capability(self, capability: str) -> None:
        self.capabilities.add(capability)
        self.touch()

    def get_namespace(self, name: str) -> Optional[ProfileNamespace]:
        return self.namespaces.get(name)

    def set_namespace(self, namespace: ProfileNamespace) -> None:
        self.namespaces[namespace.name] = namespace
        self.touch()

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
