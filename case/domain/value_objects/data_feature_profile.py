from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel,Field, field_validator
from case.domain.enums.data_feature_type import DataFeatureType
from case.domain.enums.data_type import DataType
from case.domain.value_objects.data_profile_stats import BaseStats
from case.domain.value_objects.profile_namespace import ProfileNamespace


class DataFeatureProfile (BaseModel) :
    """
    Profile of a single feature within the dataset.
        
    Attributes:
        name: Feature name (must match the key in DataProfile).
        feature_stats: Type‑specific statistics (numeric, categorical, etc.).
    """
    name : str = Field (min_length=1)
    feature_stats : BaseStats
    dtype : Any
    inferred_semantic_type : DataFeatureType = DataFeatureType.UNKNOWN

    namespaces : Dict[str, ProfileNamespace] = Field(default_factory=dict)
    capabilities : Set[str]

    computed_at : Optional[datetime] = Field(default_factory= datetime.now)
    updated_at : Optional[datetime] = None

    def touch (self) -> None:
        self.updated_at = datetime.now()