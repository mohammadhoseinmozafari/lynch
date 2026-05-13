
from __future__ import annotations
from typing import List, Optional, Dict, Set, Union
from pydantic import BaseModel, Field, field_validator
from core.case.dataset.common import  DataType, DataFeatureType

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


    
class FeatureSchema (BaseModel) :
    """
    Schema of a single feature/target  within a dataset.
    Attributes:
        name: Name of the feature/target.
        human_readable_name: raw name might not be human readable (e.g "x_1", "y_1").
        Providing a human readable name will later be useful and reduces the pain.
        dtype: Data type of the feature/target. (e.g float64)
        unconstrained: Set to True when a numeric feature has no meaningful bounds (e.g., a ratio that can be arbitrarily large).
        expected_range: Expected range (stats) of the feature/target. 
        actionable: Whether this feature can be changed in production (for counterfactual generation). E.g., age is not actionable, credit_limit is.
        description: Business meaning of the feature. Strongly encouraged; missing descriptions lower the interpretability_score.

    """
    name : str = Field(min_length=1, max_length=100)
    human_readable_name : str = Field(min_length=1, max_length=100)
    dtype : DataType 
    feature_type : DataFeatureType 
    unconstrained: Optional[bool] = Field(False)
    expected_range : Optional[Union[NumericRange, CategoricalRange]] = Field(None)
    actionable : Optional[bool] = Field(True)
    description : Optional[str] = Field(max_length=250, default="Feature description") 

    @field_validator('expected_range')
    @classmethod
    def check_expected_range_type_match(cls, v, info):
        """
        Validates that the expected_range type matches the declared feature_type.

        - If feature_type is NUMERIC, expected_range must be NumericRange (or None).
        - If feature_type is CATEGORICAL, expected_range must be CategoricalRange (or None).

        This prevents a NUMERIC feature from accidentally carrying a CategoricalRange
        and vice versa, which would corrupt counterfactual constraints and explanation baselines.
        """
        if v is None:
            return v

        # Access the feature_type from the values being validated
        declared_feature_type = info.data.get('feature_type')

        if declared_feature_type is None:
            # feature_type hasn't been validated yet or is missing; Pydantic will catch missing required fields
            return v

        if declared_feature_type == DataFeatureType.NUMERIC:
            if not isinstance(v, NumericRange):
                raise ValueError(
                    f"feature_type is '{DataFeatureType.NUMERIC.value}', but expected_range "
                    f"is a {type(v).__name__}. For numeric features, expected_range must be a NumericRange."
                )

        elif declared_feature_type == DataFeatureType.CATEGORICAL:
            if not isinstance(v, CategoricalRange):
                raise ValueError(
                    f"feature_type is '{DataFeatureType.CATEGORICAL.value}', but expected_range "
                    f"is a {type(v).__name__}. For categorical features, expected_range must be a CategoricalRange."
                )

        # If feature_type is something else (future types like DATETIME, TEXT_LENGTH),
        # we don't enforce a specific range type yet.

        return v


    @field_validator('expected_range')
    @classmethod
    def check_range_consistency_with_unconstrained(cls, v, info):
        """
        If unconstrained is True, a numeric feature should not have an expected_range.

        An unconstrained feature has no meaningful bounds, so providing a range is contradictory.
        """
        if v is None:
            return v

        is_unconstrained = info.data.get('unconstrained', False)
        if is_unconstrained and isinstance(v, NumericRange):
            if v.min is not None or v.max is not None:
                raise ValueError(
                    "Feature is marked as unconstrained, but expected_range has min/max values. "
                    "Either remove the range or set unconstrained=False."
                )

        return v
            
class NumericRange (BaseModel):
    """
    Numeric range is a simple range (min,max). 
    """
    min : Optional[float] = None
    max : Optional[float] = None

    

class CategoricalRange (BaseModel) :
    """
    For categorical features we cannot define a numerical range. 
    The categorical range is the possible values of a categorical feature.
    """
    possible_values : List[str]

