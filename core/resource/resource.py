from datetime import datetime
from enum import Enum
from typing import Any
import uuid
import pandas as pd
from sklearn.base import BaseEstimator
from pydantic import BaseModel, Field

class ResourceType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"


class Resource(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    resource_type: ResourceType
    value: Any
    created_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Resource":
        resource_id = str(uuid.uuid4())

        return cls(
            id=resource_id,
            name=f"dataset-{resource_id[:8]}",
            resource_type=ResourceType.DATASET,
            value=df,
        )
    @classmethod
    def from_model(cls, model: BaseEstimator ) -> "Resource":
        resource_id = str(uuid.uuid4())

        return cls(
            id=resource_id,
            name=f"model-{resource_id[:8]}",
            resource_type=ResourceType.MODEL,
            value=model,
        )
    
