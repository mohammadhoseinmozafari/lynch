from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from ulid import ulid

from core.profilers.missingness.models import SegmentProfile
from core.profilers.missingness.profiler import ProfilerType


class MissingnessSegmentProfiler(ABC):
    id: str
    profiler_type: ProfilerType

    @abstractmethod
    def profile_segments(
        self,
        df: pd.DataFrame,
        target: Union[str, Sequence[bool], Sequence[int], pd.Series],
    ) -> List[SegmentProfile]:
        raise NotImplementedError


@dataclass(frozen=True)
class _FeatureMetadata:
    name: str
    kind: str
    categories: List[Any] | None = None


class PandasMissingnessSegmentProfiler(MissingnessSegmentProfiler):
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type = ProfilerType.PANDAS
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def profile_segments(
        self,
        df: pd.DataFrame,
        target: Union[str, Sequence[bool], Sequence[int], pd.Series],
    ) -> List[SegmentProfile]:
        if df.empty:
            return []

        feature_df, target_mask = self._resolve_target(df, target)
        total_rows = len(feature_df)

        if total_rows == 0:
            return []

        total_missing_count = int(target_mask.sum())
        global_missing_rate = float(target_mask.mean()) if total_rows > 0 else 0.0

        if feature_df.shape[1] == 0:
            return [
                SegmentProfile(
                    rules=[],
                    row_count=total_rows,
                    coverage=1.0,
                    missing_count=total_missing_count,
                    missing_rate=global_missing_rate,
                    global_missing_rate=global_missing_rate,
                    contribution=1.0 if total_missing_count > 0 else 0.0,
                    lift=1.0 if global_missing_rate > 0 else 0.0,
                )
            ]

        feature_matrix, metadata = self._transform_features(feature_df)
        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        model.fit(feature_matrix, target_mask.astype(int).to_numpy())

        leaf_ids = model.apply(feature_matrix)
        leaf_rules = self._extract_leaf_rules(model, metadata)

        segment_profiles: List[SegmentProfile] = []
        for leaf_id in np.unique(leaf_ids):
            leaf_indices = np.flatnonzero(leaf_ids == leaf_id)
            row_count = int(len(leaf_indices))
            missing_count = int(target_mask.iloc[leaf_indices].sum())
            missing_rate = missing_count / row_count if row_count > 0 else 0.0
            coverage = row_count / total_rows if total_rows > 0 else 0.0
            contribution = (
                missing_count / total_missing_count if total_missing_count > 0 else 0.0
            )
            lift = (
                missing_rate / global_missing_rate if global_missing_rate > 0 else 0.0
            )

            segment_profiles.append(
                SegmentProfile(
                    rules=leaf_rules.get(int(leaf_id), []),
                    row_count=row_count,
                    coverage=coverage,
                    missing_count=missing_count,
                    missing_rate=missing_rate,
                    global_missing_rate=global_missing_rate,
                    contribution=contribution,
                    lift=lift,
                )
            )

        return segment_profiles

    def _resolve_target(
        self,
        df: pd.DataFrame,
        target: Union[str, Sequence[bool], Sequence[int], pd.Series],
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(target, str):
            if target not in df.columns:
                raise KeyError(f"Target column '{target}' not found in DataFrame")
            feature_df = df.drop(columns=[target])
            target_mask = df[target].isna()
            return feature_df, target_mask.astype(bool)

        if isinstance(target, pd.Series):
            target_series = pd.Series(target.to_numpy(), index=df.index)
        else:
            target_series = pd.Series(list(target), index=df.index)

        if len(target_series) != len(df):
            raise ValueError("Target mask length must match the DataFrame row count")

        if target_series.isna().any():
            target_series = target_series.fillna(False)

        return df, target_series.astype(bool)

    def _transform_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, List[_FeatureMetadata]]:
        numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_columns = [
            column
            for column in df.columns
            if column not in numeric_columns
        ]

        metadata: List[_FeatureMetadata] = []
        parts: List[np.ndarray] = []

        for column in numeric_columns:
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            median_value = numeric_series.median()
            fill_value = 0.0 if pd.isna(median_value) else float(median_value)
            filled = numeric_series.fillna(fill_value).to_numpy(dtype=float).reshape(-1, 1)
            parts.append(filled)
            metadata.append(_FeatureMetadata(name=column, kind="numeric"))

        for column in categorical_columns:
            categorical_series = df[column].astype("object").where(
                df[column].notna(), other="__missing__"
            )
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            encoded = encoder.fit_transform(categorical_series.to_frame()).astype(float)
            parts.append(encoded)
            metadata.append(
                _FeatureMetadata(
                    name=column,
                    kind="categorical",
                    categories=list(encoder.categories_[0]),
                )
            )

        if not parts:
            return np.empty((len(df), 0)), metadata

        return np.hstack(parts), metadata

    def _extract_leaf_rules(
        self,
        model: DecisionTreeClassifier,
        metadata: List[_FeatureMetadata],
    ) -> Dict[int, List[str]]:
        tree = model.tree_
        leaf_rules: Dict[int, List[str]] = {}

        def recurse(node_id: int, path_rules: List[str]) -> None:
            feature_index = tree.feature[node_id]
            if feature_index == _tree.TREE_UNDEFINED:
                leaf_rules[node_id] = list(path_rules)
                return

            feature = metadata[feature_index]
            threshold = float(tree.threshold[node_id])

            left_rule, right_rule = self._build_branch_rules(feature, threshold)

            recurse(tree.children_left[node_id], path_rules + [left_rule])
            recurse(tree.children_right[node_id], path_rules + [right_rule])

        recurse(0, [])
        return leaf_rules

    def _build_branch_rules(
        self,
        feature: _FeatureMetadata,
        threshold: float,
    ) -> tuple[str, str]:
        if feature.kind == "numeric":
            return (
                f"{feature.name} <= {threshold:.6g}",
                f"{feature.name} > {threshold:.6g}",
            )

        categories = feature.categories or []
        cutoff = int(np.floor(threshold))
        left_categories = [
            category
            for code, category in enumerate(categories)
            if code <= cutoff
        ]
        right_categories = [
            category
            for code, category in enumerate(categories)
            if code > cutoff
        ]

        left_values = [self._format_category_value(value) for value in left_categories]
        right_values = [self._format_category_value(value) for value in right_categories]

        if threshold >= -0.5:
            left_values = ["__unknown__"] + left_values

        return (
            f"{feature.name} in {{{', '.join(left_values)}}}" if left_values else f"{feature.name} in {{}}",
            f"{feature.name} in {{{', '.join(right_values)}}}" if right_values else f"{feature.name} in {{}}",
        )

    @staticmethod
    def _format_category_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return str(value)