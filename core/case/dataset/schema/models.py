
from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from core.case.artifact import DataArtifactPointer
from core.case.dataset.common import DataSplitName, DataType, DataFeatureType, TaskType

class DataSchema (BaseModel) :
    """
    Schema of a dataset. Used to track structrul changes of the dataset
    
    Attributes:
        splits: Splits of the data with their corresponding split schema.
        task_type: Shows what task type will the dataset be used for. (e.g regression, classification)
    """
    splits: Dict[DataSplitName, DataSplitSchema]
    task_type : Optional[TaskType] = Field(default=None)

    def get_split(self, split_name : str) -> Optional[DataSplitSchema]:
        enum_key = DataSplitName(split_name)
        return self.splits.get(enum_key)
class DataSplitSchema (BaseModel):
    """
    Schema of a single data split (e.g train set)
    Attributes:
        split_name: Name of the split (currently supports train , val and test)
        split_artifact: Pointer to the data split artifact. 
        features: List of features of the data split.
        targets: List of targets of the data split
        metadata_columns: Metadata columns that won't be used for model training. (e.g 'id')
        
    """
    split_name: DataSplitName
    split_artifact: DataArtifactPointer
    features: Dict[str, FeatureSchema] = Field(min_length=1) 
    targets: Optional[Dict[str, FeatureSchema]]= Field(default=None)
    metadata_columns: Optional[List[str]] = Field(default=None)

    @property 
    def has_target_variables(self) ->bool:
        return self.targets is not None
    
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
        expected_range: Expected range (stats) of the feature/target. 
            For example we might calculate feature statistics of a train set and we know that the range for a specific feature is (50, 100),
            then we can expect the same range in the validation set. The corresponding feature in val set will have expected_range = (50,100).
        description: Description of the feature/target. This is optional, but crucial for interpretability.
    """
    name : str = Field(min_length=1, max_length=255, pattern='^[a-zA-Z_][a-zA-Z0-9_]*$')
    human_readable_name : str = Field(min_length=1)
    dtype : DataType 
    expected_range : Optional[BaseRange] = None
    description : Optional[str] = Field(max_length=150, default="Feature description") 

class BaseRange (BaseModel) :
    feature_type : DataFeatureType

class NumericRange (BaseRange):
    """
    Numeric range is a simple range (min,max). 
    """
    feature_type : DataFeatureType = Field(DataFeatureType.NUMERIC, frozen=True)
    min : Optional[float] = None
    max : Optional[float] = None

class CategoricalRange (BaseRange) :
    """
    For categorical features we cannot define a numerical range. 
    The categorical range is the possible values of a categorical feature.
    """
    feature_type : DataFeatureType = Field(default= DataFeatureType.CATEGORICAL, frozen= True)
    possible_values : List[str]

