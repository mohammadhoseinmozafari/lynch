from __future__ import annotations

from typing import Any, Dict, Literal, cast

import pandas as pd
from ulid import ulid

from core.profile import DatasetProfiler
from core.profile.profiler import ProfilerType
from .models import (
    ColumnMissingRateProfile,
    MissingnessDistributionProfile,
    RowsMissingRateProfile,
)


class ColumnMissingRateProfiler(DatasetProfiler):
    def __init__(self) -> None:
        super().__init__()

    def profile(self, df: pd.DataFrame) -> Dict[str, ColumnMissingRateProfile]:
        total_rows = len(df)
        missing_counts = df.isna().sum()
        non_missing_counts = total_rows - missing_counts

        return {
            column: ColumnMissingRateProfile(
                column_name=column,
                missing_count=int(missing_counts[column]),
                non_missing_count=int(non_missing_counts[column]),
                total_count=total_rows,
                missing_rate=float(missing_counts[column] / total_rows)
                if total_rows > 0
                else 0.0,
            )
            for column in df.columns
        }



class RowsMissingRateProfiler(DatasetProfiler):
    def __init__(self, sample_size: int = 10, threshold: float = 0.8) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type = ProfilerType.DATASET
        self.sample_size = sample_size
        self.threshold = threshold

    def profile(self, df: pd.DataFrame) -> RowsMissingRateProfile:
        rows_missing_rates = df.isna().mean(axis=1)

        full_missing_rows_profile = self.profile_full_missing_rows(
            rows_missing_rates,
            self.sample_size,
        )
        high_missing_rate_rows_profile = self.profile_high_missing_rate_rows(
            rows_missing_rates,
            threshold=self.threshold,
            sample_size=self.sample_size,
        )

        total_rows = len(df)

        return RowsMissingRateProfile(
            total_rows=total_rows,
            full_missing_rows_count=full_missing_rows_profile["count"],
            full_missing_rows_rate=full_missing_rows_profile["count"] / total_rows
            if total_rows > 0
            else 0.0,
            full_missing_rows_sample_indices=full_missing_rows_profile[
                "sample_indices"
            ],
            high_missing_rate_rows_count=high_missing_rate_rows_profile["count"],
            high_missing_rate_rows_rate=high_missing_rate_rows_profile["count"]
            / total_rows
            if total_rows > 0
            else 0.0,
            high_missing_rate_rows_sample_indices=high_missing_rate_rows_profile[
                "sample_indices"
            ],
            sample_size=self.sample_size,
        )
    
    def profile_full_missing_rows(
        self,
        rows_missing_rates: pd.Series,
        sample_size: int,
    ) -> Dict[str, Any]:
        fully_missing_rows = rows_missing_rates[rows_missing_rates == 1.0]
        fully_missing_count = int(len(fully_missing_rows))

        if fully_missing_count == 0:
            return {"count": 0, "sample_indices": []}

        sample_indices = fully_missing_rows.sample(
            n=min(fully_missing_count, sample_size),
            random_state=42,
        ).index.tolist()

        return {"count": fully_missing_count, "sample_indices": sample_indices}

    def profile_high_missing_rate_rows(
        self,
        rows_missing_rates: pd.Series,
        threshold: float,
        sample_size: int,
    ) -> Dict[str, Any]:
        high_missing_rate_rows = rows_missing_rates[
            (rows_missing_rates >= threshold) & (rows_missing_rates < 1.0)
        ]
        high_missing_count = int(len(high_missing_rate_rows))

        if high_missing_count == 0:
            return {"count": 0, "sample_indices": []}

        sample_indices = high_missing_rate_rows.sample(
            n=min(high_missing_count, sample_size),
            random_state=42,
        ).index.tolist()

        return {"count": high_missing_count, "sample_indices": sample_indices}


class DistributionMissingRateProfiler(DatasetProfiler):
    def __init__(self, axis: Literal["index", "columns"] = "columns") -> None:
        super().__init__()
        self.axis = axis

    def profile(self, df: pd.DataFrame) -> MissingnessDistributionProfile:
        missing_rates = df.isna().mean(axis=cast(Any, self.axis))

        return MissingnessDistributionProfile(
            mean_missing_rate=float(missing_rates.mean()),
            median_missing_rate=float(missing_rates.median()),
            std_missing_rate=float(missing_rates.std()),
            min_missing_rate=float(missing_rates.min()),
            max_missing_rate=float(missing_rates.max()),
            p90_missing_rate=float(missing_rates.quantile(0.90)),
            p95_missing_rate=float(missing_rates.quantile(0.95)),
            p99_missing_rate=float(missing_rates.quantile(0.99)),
        )


class MissingRateProfiler(DatasetProfiler):
    def __init__(self, sample_size: int = 10, threshold: float = 0.8) -> None:
        super().__init__()

        self._column_profiler = ColumnMissingRateProfiler()
        self._rows_profiler = RowsMissingRateProfiler(
            sample_size=sample_size,
            threshold=threshold,
        )
        self._rows_distribution_profiler = DistributionMissingRateProfiler(
            axis="columns"
        )
        self._columns_distribution_profiler = DistributionMissingRateProfiler(
            axis="index"
        )

    def profile_columns(self, df: pd.DataFrame) -> Dict[str, ColumnMissingRateProfile]:
        return self._column_profiler.profile(df)

    def profile_rows(self, df: pd.DataFrame, sample_size: int) -> RowsMissingRateProfile:
        return RowsMissingRateProfiler(
            sample_size=sample_size,
            threshold=self._rows_profiler.threshold,
        ).profile(df)

    def profile_distribution(self, df: pd.DataFrame) -> MissingnessDistributionProfile:
        return self._rows_distribution_profiler.profile(df)

    def profile_rows_distribution(
        self,
        df: pd.DataFrame,
    ) -> MissingnessDistributionProfile:
        return self._rows_distribution_profiler.profile(df)

    def profile_columns_distribution(
        self,
        df: pd.DataFrame,
    ) -> MissingnessDistributionProfile:
        return self._columns_distribution_profiler.profile(df)

    