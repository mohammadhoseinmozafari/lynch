from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from ulid import ulid

from case.domain.enums.data_feature_type import DataFeatureType
from case.domain.enums.data_type import DataType
from case.domain.value_objects.data_feature_profile import DataFeatureProfile
from case.domain.value_objects.data_feature_schema import FeatureSchema
from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.data_profile_range import NumericRange
from case.domain.value_objects.data_profile_stats import CategoricalStats, NumericStats
from case.domain.value_objects.data_schema import DataSchema
from case.domain.value_objects.dataset_artifact import InMemoryDataArtifactPointer
from case.domain.value_objects.dataset_binding import DatasetBinding


class ProfilerType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"


class DatasetProfiler(ABC):
    """Base class for profilers that inspect a pandas dataset."""

    capability: str 

    def __init__(self) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type: ProfilerType = ProfilerType.DATASET

    @abstractmethod
    def profile(self, df: pd.DataFrame) -> Any:
        pass

    def profile_binding(self, binding: DatasetBinding) -> Any:
        """Profile an existing binding and record this profiler's capability."""
        result = self.profile(binding.artifact_pointer.resolve())
        binding.dataset_profile.add_capability(self.capability_name)
        return result

    @property
    def capability_name(self) -> str:
        return self.capability



class BaseDatasetProfiler(DatasetProfiler):
    """Create the initial binding, schema, and profile for a DataFrame."""

    capability = "profile.base"

    def __init__(self, dataset_name: Optional[str] = None) -> None:
        super().__init__()
        self.dataset_name = dataset_name

    def profile(self, df: pd.DataFrame) -> DatasetBinding:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.columns.empty:
            raise ValueError("df must contain at least one column")
        if not all(isinstance(column, str) and column for column in df.columns):
            raise ValueError("all DataFrame columns must be non-empty strings")
        if not df.columns.is_unique:
            raise ValueError("DataFrame columns must be unique")

        feature_schemas: Dict[str, FeatureSchema] = {}
        feature_profiles: Dict[str, DataFeatureProfile] = {}
        

        for name in df.columns:
            series = df[name]
            data_type, feature_type = self._infer_types(series)
            feature_schemas[name] = FeatureSchema(
                name=name,
                human_readable_name=name,
                dtype=data_type,
                feature_type=feature_type,
            )
            feature_profiles[name] = DataFeatureProfile(
                name=name,
                feature_stats=self._profile_series(series, feature_type),
                dtype=series.dtype,
                inferred_semantic_type=feature_type,
                capabilities={self.capability},
            )

        data_profile = DataProfile(feature_profiles=feature_profiles)
        data_profile.add_capability(self.capability)

        return DatasetBinding(
            dataset_name=self.dataset_name,
            artifact_pointer=InMemoryDataArtifactPointer.from_dataframe(df),
            dataset_schema=DataSchema(features=feature_schemas),
            dataset_profile=data_profile,
        )

    @staticmethod
    def _infer_types(series: pd.Series) -> Tuple[DataType, DataFeatureType]:
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOL, DataFeatureType.BOOLEAN
        if pd.api.types.is_integer_dtype(dtype):
            return DataType.INT64, DataFeatureType.NUMERIC
        if pd.api.types.is_numeric_dtype(dtype):
            return DataType.FLOAT64, DataFeatureType.NUMERIC
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return DataType.DATETIME64, DataFeatureType.DATETIME
        if isinstance(dtype, pd.CategoricalDtype):
            return DataType.TEXT, DataFeatureType.CATEGORICAL
        return DataType.TEXT, DataFeatureType.CATEGORICAL

    @staticmethod
    def _profile_series(series: pd.Series, feature_type: DataFeatureType) -> Any:
        if feature_type == DataFeatureType.NUMERIC:
            non_null = series.dropna()
            if non_null.empty:
                return NumericStats(
                    range=NumericRange(),
                    mean=float("nan"),
                    std=float("nan"),
                    quartiles=(float("nan"),) * 3,
                )

            quartiles = non_null.quantile([0.25, 0.5, 0.75])
            return NumericStats(
                range=NumericRange(
                    min=float(non_null.min()),
                    max=float(non_null.max()),
                ),
                mean=float(non_null.mean()),
                std=float(non_null.std()),
                quartiles=tuple(float(value) for value in quartiles),
            )

        counts = series.value_counts(dropna=False, normalize=True).head(10)
        top_values = {
            BaseDatasetProfiler._display_value(value): float(rate)
            for value, rate in counts.items()
        }
        return CategoricalStats(
            cardinality=int(series.nunique(dropna=True)),
            top_values=top_values,
        )

    @staticmethod
    def _display_value(value: Any) -> str:
        return "<missing>" if pd.isna(value) else str(value)
