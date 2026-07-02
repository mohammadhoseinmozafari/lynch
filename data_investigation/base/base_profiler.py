from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from case.domain.enums.data_feature_type import DataFeatureType
from case.domain.value_objects.data_feature_profile import DataFeatureProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from core.context import InvestigationContext
from core.profile import DatasetProfiler


class DataframeValidator:
    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Holmes supports only pandas DataFrames currently ")
        if df.columns.empty:
            raise ValueError("df must contain at least one column")
        if not all(isinstance(column, str) and column for column in df.columns):
            raise ValueError("all DataFrame columns must be non-empty strings")
        if not df.columns.is_unique:
            raise ValueError("DataFrame columns must be unique")


class BaseDatasetProfiler(DatasetProfiler):
    """Compute base per-feature statistics without mutating the profile."""

    capability = "profile.base"
    requires: set[str] = set()
    provides = {capability}

    def __init__(self, dataset_name: Optional[str] = None) -> None:
        super().__init__()
        self.dataset_name = dataset_name

    def profile(self, ctx: InvestigationContext) -> ProfileNamespace:
        df = ctx.get_dataframe()
        DataframeValidator.validate(df)
        feature_profiles: Dict[str, DataFeatureProfile] = {}
        total_rows = len(df)
        for name in df.columns:
            series = df[name]
            feature_type = self._infer_types(series)
            feature_profiles[name] = DataFeatureProfile(
                name=name,
                dtype=series.dtype,
                inferred_semantic_type=feature_type,
            )

        return ProfileNamespace(
            name=self.capability,
            metrics={
                "feature_profiles": feature_profiles,
                "total_rows": total_rows,
            },
        )

    @staticmethod
    def _infer_types(series: pd.Series) -> DataFeatureType:
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return DataFeatureType.BOOLEAN
        if pd.api.types.is_integer_dtype(dtype):
            return DataFeatureType.NUMERIC
        if pd.api.types.is_numeric_dtype(dtype):
            return DataFeatureType.NUMERIC
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return DataFeatureType.DATETIME
        if isinstance(dtype, pd.CategoricalDtype):
            return DataFeatureType.CATEGORICAL
        return DataFeatureType.CATEGORICAL
