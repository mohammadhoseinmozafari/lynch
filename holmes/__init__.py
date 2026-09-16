"""Holmes notebook-native public API (MVP)."""

from typing import Optional
import pandas as pd
from sklearn.base import BaseEstimator
from api.dataset import Dataset
from api.model import Model
from .runtime import get_runtime
from core.resource import ResourceValidator

def dataset(
        df: pd.DataFrame, 
        *,
        name : Optional[str] = None
) -> Dataset:
    
    ResourceValidator.dataset(df)
    
    return get_runtime().register_dataset(
        df,
        name = name
    )

def model(
    model: BaseEstimator,
    *,
    name: str | None = None,
) -> Model:
    
    ResourceValidator.model(model)

    return get_runtime().register_model(
        model,
        name=name,
    )

__all__ = [ "Dataset",
            "Model",
            "dataset",
            "model",
            
            ]


