from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from numbers import Integral, Real
from typing import Any

import pandas as pd
from ulid import ulid

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from core.profile import DatasetProfiler
from core.profile.profiler import ProfilerType



class ColumnMissingRateProfiler(DatasetProfiler):
    """Compute per-column missingness metrics."""

    capability = "missingness.column_rates"
    requires = {"profile.base"}
    provides = {capability}

    def profile(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
    ) -> ProfileNamespace:
        
        total_rows = len(df)
        missing_counts = df.isna().sum()
        non_missing_counts = total_rows - missing_counts
        missing_rates = (
            missing_counts / total_rows
            if total_rows > 0
            else pd.Series(0.0, index=df.columns)
        )

        return ProfileNamespace(
            name=self.capability,
            metrics={
                "missing_rate_by_column": {
                    column: float(missing_rates[column]) for column in df.columns
                },
                "missing_count_by_column": {
                    column: int(missing_counts[column]) for column in df.columns
                },
                "non_missing_count_by_column": {
                    column: int(non_missing_counts[column]) for column in df.columns
                },
                "total_rows": total_rows,
            },
        )
    

class RowsMissingRateProfiler(DatasetProfiler):
    """Compute row-level missingness metrics."""

    capability = "missingness.row_rates"
    requires = {"profile.base"}
    provides = {capability}

    def __init__(self, sample_size: int = 10, threshold: float = 0.8) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type = ProfilerType.DATASET
        self.sample_size = self._validate_sample_size(sample_size)
        self.threshold = self._validate_threshold(threshold)

    def profile(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
    ) -> ProfileNamespace:
        
        rows_missing_rates = df.isna().mean(axis=1)
        fully_missing_rows = rows_missing_rates[rows_missing_rates == 1.0]
        high_missing_rows = rows_missing_rates[
            (rows_missing_rates >= self.threshold) & (rows_missing_rates < 1.0)
        ]
        total_rows = len(df)
        full_count = int(len(fully_missing_rows))
        high_count = int(len(high_missing_rows))

        return ProfileNamespace(
            name=self.capability,
            metrics={
                "full_missing_rows_count": full_count,
                "full_missing_rows_rate": (
                    full_count / total_rows if total_rows > 0 else 0.0
                ),
                "high_missing_rows_count": high_count,
                "high_missing_rows_rate": (
                    high_count / total_rows if total_rows > 0 else 0.0
                ),
                "sample_indices": {
                    "full_missing_rows": self._sample_indices(
                        fully_missing_rows, self.sample_size
                    ),
                    "high_missing_rows": self._sample_indices(
                        high_missing_rows, self.sample_size
                    ),
                },
                "threshold": self.threshold,
                "sample_size": self.sample_size,
                "total_rows": total_rows,
            },
        )

    @staticmethod
    def _validate_sample_size(sample_size: int) -> int:
        if (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, Integral)
            or sample_size < 0
        ):
            raise ValueError("sample_size must be a non-negative integer")
        return int(sample_size)

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be a finite number in [0.0, 1.0]")
        return float(threshold)

    @staticmethod
    def _sample_indices(rows: pd.Series, sample_size: int) -> list[Any]:
        if rows.empty or sample_size == 0:
            return []
        return rows.sample(
            n=min(len(rows), sample_size), random_state=42
        ).index.tolist()


class ColumnDistributionMissingRateProfiler(DatasetProfiler):
    """Compute a missing-rate distribution using cached column rates if present."""

    capability = "missingness.distribution.columns"
    requires = {"missingness.column_rates"}
    provides = {capability}

    def profile(self, df: pd.DataFrame, profile: DataProfile) -> ProfileNamespace:
        column_namespace = profile.require_namespace("missingness.column_rates")
        rates = column_namespace.require_metric("missing_rate_by_column")

        validated_rates = self._validate_rates(rates, df)

        self._validate_count_consistency(column_namespace, validated_rates)

        missing_rates = pd.Series(validated_rates, dtype=float)

        return ProfileNamespace(
            name=self.capability,
            metrics=self._distribution_metrics(missing_rates),
        )

    @staticmethod
    def _validate_rates(rates: Any, df: pd.DataFrame) -> dict[Any, float]:
        if not isinstance(rates, Mapping):
            raise ValueError("missing_rate_by_column must be a mapping")
        if set(rates) != set(df.columns):
            raise ValueError("Column missing rates do not match DataFrame columns")

        validated: dict[Any, float] = {}
        for column, rate in rates.items():
            if (
                isinstance(rate, bool)
                or not isinstance(rate, Real)
                or not isfinite(float(rate))
                or not 0.0 <= float(rate) <= 1.0
            ):
                raise ValueError(f"Invalid missing rate for column {column!r}: {rate}")
            validated[column] = float(rate)
        return validated

    @staticmethod
    def _validate_count_consistency(
        namespace: ProfileNamespace,
        rates: Mapping[Any, float],
    ) -> None:
        counts = namespace.metrics.get("missing_count_by_column")
        total_rows = namespace.metrics.get("total_rows")
        if counts is None or total_rows is None:
            return
        if not isinstance(counts, Mapping):
            raise ValueError("missing_count_by_column must be a mapping")
        if isinstance(total_rows, bool) or not isinstance(total_rows, Integral):
            raise ValueError("total_rows must be a non-negative integer")
        total_rows = int(total_rows)
        if total_rows < 0:
            raise ValueError("total_rows must be a non-negative integer")
        if set(counts) != set(rates):
            raise ValueError("Column missing counts do not match missing-rate columns")

        for column, rate in rates.items():
            count = counts[column]
            if (
                isinstance(count, bool)
                or not isinstance(count, Integral)
                or not 0 <= int(count) <= total_rows
            ):
                raise ValueError(
                    f"Invalid missing count for column {column!r}: {count}"
                )
            expected_rate = int(count) / total_rows if total_rows > 0 else 0.0
            if not isclose(rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"Missing rate for column {column!r} is inconsistent with its count"
                )

    @staticmethod
    def _distribution_metrics(missing_rates: pd.Series) -> dict[str, float]:
        return {
            "mean": float(missing_rates.mean()),
            "median": float(missing_rates.median()),
            "std": float(missing_rates.std()),
            "min": float(missing_rates.min()),
            "max": float(missing_rates.max()),
            "p90": float(missing_rates.quantile(0.90)),
            "p95": float(missing_rates.quantile(0.95)),
            "p99": float(missing_rates.quantile(0.99)),
        }


class RowsDistributionMissingRateProfiler(DatasetProfiler):
    """Compute row-wise missing-rate distribution."""

    capability = "missingness.distribution.rows"
    requires = {"profile.base"}
    provides = {capability}

    def profile(self, df: pd.DataFrame, profile: DataProfile) -> ProfileNamespace:
        missing_rates = df.isna().mean(axis=1)


        return ProfileNamespace(
            name=self.capability,
            metrics=self._distribution_metrics(
                missing_rates
            ),
        )

    @staticmethod
    def _distribution_metrics(missing_rates: pd.Series) -> dict[str, float]:
        return {
            "mean": float(missing_rates.mean()),
            "median": float(missing_rates.median()),
            "std": float(missing_rates.std()),
            "min": float(missing_rates.min()),
            "max": float(missing_rates.max()),
            "p90": float(missing_rates.quantile(0.90)),
            "p95": float(missing_rates.quantile(0.95)),
            "p99": float(missing_rates.quantile(0.99)),
        }
