
from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from core.case.artifact import DataArtifactPointer
from core.case.dataset import DataSplitName, DataType, DataFeatureType, TaskType

class DataSchema (BaseModel) :
    """
    Schema of a dataset. Used to track structrul changes of the dataset
    
    Attributes:
        splits: Splits of the data with their corresponding split schema.
        task_type: Shows what task type will the dataset be used for. (e.g regression, classification)
    """
    splits: Dict[DataSplitName, DataSplitSchema]
    task_type : Optional[TaskType] = Field(default=None)

class DataSplitSchema (BaseModel):
    """
    Schema of a single data split (e.g train set)
    Attributes:
        split_name: Name of the split (currently supports train , val and test)
        split_artifact: Pointer to the data split artifact. 
        features: List of features of the data split.
        metadata_columns: Metadata columns that won't be used for model training. (e.g 'id')
        target: Name of the target column.
    """
    split_name: DataSplitName
    split_artifact: DataArtifactPointer
    features: List[FeatureSchema] = Field(min_length=1) 
    metadata_columns: Optional[List[str]] = Field(default=None)
    target: Optional[str]= Field(default=None)

    
class FeatureSchema (BaseModel) :
    """
    Schema of a single feature within a dataset.
    Attributes:
        name: Name of the feature.
        human_readable_name: raw name might not be human readable (e.g "x_1", "feature_1").
            Providing a human readable name will later be useful and reduces the pain.
        dtype: Data type of the feature. (e.g float64)
        expected_range: Expected range (stats) of the feature. 
            For example we might calculate statistics of a train set and we know that the range for a specific feature is (50, 100),
            then we can expect the same range in the validation set
        description: Description of the feature. This is optional, but crucial for interpretability.
        is_target: Shows if the feature is a target or not.
    """
    name : str = Field(min_length=1, max_length=255, pattern='^[a-zA-Z_][a-zA-Z0-9_]*$')
    human_readable_name : str = Field(min_length=1)
    dtype : DataType 
    expected_range : Optional[BaseRange] = None
    description : Optional[str] = Field(max_length=150, default="Feature description") 
    is_target : bool = False

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

