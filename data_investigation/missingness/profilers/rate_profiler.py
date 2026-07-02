from __future__ import annotations

from typing import Any

import pandas as pd
from ulid import ulid

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from core.profile import DatasetProfiler
from core.profile.profiler import ProfilerType


class ColumnMissingRateProfiler(DatasetProfiler):
    """Add per-column missingness metrics to a shared data profile."""

    capability = "missingness.column_rates"
    requires = {"profile.base"}
    provides = {capability}

    def profile(self, df: pd.DataFrame, profile: DataProfile) -> None:

        total_rows = len(df)
        missing_counts = df.isna().sum()
        non_missing_counts = total_rows - missing_counts
        missing_rates = (
            missing_counts / total_rows
            if total_rows > 0
            else pd.Series(0.0, index=df.columns)
        )

        profile.set_namespace(
            ProfileNamespace(
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
        )
        profile.add_capability(self.capability)


class RowsMissingRateProfiler(DatasetProfiler):
    """Add row-level missingness metrics to a shared data profile."""

    capability = "missingness.row_rates"
    requires = {"profile.base"}
    provides = {capability}

    def __init__(self, sample_size: int = 10, threshold: float = 0.8) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type = ProfilerType.DATASET
        self.sample_size = sample_size
        self.threshold = threshold

    def profile(self, df: pd.DataFrame, profile: DataProfile) -> None:


        rows_missing_rates = df.isna().mean(axis=1)
        fully_missing_rows = rows_missing_rates[rows_missing_rates == 1.0]
        high_missing_rows = rows_missing_rates[
            (rows_missing_rates >= self.threshold) & (rows_missing_rates < 1.0)
        ]
        total_rows = len(df)
        full_count = int(len(fully_missing_rows))
        high_count = int(len(high_missing_rows))

        profile.set_namespace(
            ProfileNamespace(
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
        )
        profile.add_capability(self.capability)

    @staticmethod
    def _sample_indices(rows: pd.Series, sample_size: int) -> list[Any]:
        if rows.empty:
            return []
        return rows.sample(
            n=min(len(rows), sample_size), random_state=42
        ).index.tolist()


class DistributionMissingRateProfiler(DatasetProfiler):
    """Add a missing-rate distribution, preferring cached column rates."""

    capability = "missingness.distribution"
    requires=  {"profile.base"}
    requires_any = {"missingness.column_rates", "missingness.row_rates"}
    provides = {capability}

    def profile(self, df: pd.DataFrame, profile: DataProfile) -> None:


        column_namespace = profile.get_namespace("missingness.column_rates")
        if column_namespace is not None:
            rates = column_namespace.metrics.get("missing_rate_by_column", {})
            missing_rates = pd.Series(rates, dtype=float)
        else:
            missing_rates = df.isna().mean(axis=1)

        profile.set_namespace(
            ProfileNamespace(
                name=self.capability,
                metrics={
                    "mean": float(missing_rates.mean()),
                    "median": float(missing_rates.median()),
                    "std": float(missing_rates.std()),
                    "min": float(missing_rates.min()),
                    "max": float(missing_rates.max()),
                    "p90": float(missing_rates.quantile(0.90)),
                    "p95": float(missing_rates.quantile(0.95)),
                    "p99": float(missing_rates.quantile(0.99)),
                },
            )
        )
        profile.add_capability(self.capability)
