
from __future__ import annotations
from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field, field_validator
from core.case.domain.value_objects.data_feature_schema import FeatureSchema

class DataSchema (BaseModel) :
    """
    Schema of a dataset. Used to track structural changes of the dataset
    
    Attributes:
        features: List of features of the dataset.
        targets: List of targets of the dataset.
        metadata_columns: Metadata columns that won't be used for model training. (e.g 'id')
    """
    features: Dict[str, FeatureSchema] = Field(min_length=1) 
    targets: Optional[Dict[str, FeatureSchema]]= Field(default=None)
    metadata_columns: Optional[List[str]] = Field(default=None)


    def get_features_names(self) -> Set[str]:
        return set(self.features.keys())

    def get_targets_names (self) -> Optional[Set[str]]:
        return set(self.targets.keys()) if self.targets else None
    
    @field_validator('features','targets')
    def check_feature_names_match_keys(cls, v) -> None:
        """Ensures that the 'name' attribute of each FeatureSchema matches it's key in the dictionary."""
        if not v:
            return v
        for dict_name, feature_schema in v.items():
            if feature_schema.name != dict_name:
                raise ValueError(f"Feature name mismatch: Key '{dict_name}' doesn't match FeatureSchema's internal name '{feature_schema.name}'")
        return v
