from __future__ import annotations
from typing import  Optional
from pydantic import BaseModel, Field
from case.domain.enums.data_type import DataType
from case.domain.enums.data_feature_type import DataFeatureType
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

    unconstrained: Optional[bool] = False
    actionable : Optional[bool] = True
    
    description : Optional[str] = Field(max_length=250, default="Feature description") 
