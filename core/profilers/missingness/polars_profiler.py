
from typing import Any, Dict
import polars as pl
import numpy as np
from core.profilers.missingness.models import (
    ColumnMissingRateProfile,
    RowsMissingRateProfile,
    MissingnessDistributionProfile
    )
from core.profilers.missingness.profiler import MissingRateProfiler, ProfilerType
from ulid import ulid

class PolarsMissingRateProfiler(MissingRateProfiler[pl.DataFrame]):

    def __init__(self) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type = ProfilerType.POLARS

    def profile_columns(self, df: pl.DataFrame) -> Dict[str, ColumnMissingRateProfile]:

        total_rows = df.height
        null_counts = df.null_count()  # 1-row DataFrame with null count per column

        profile = {}

        for column in df.columns:

            missing_count = int(null_counts[column][0])
            non_missing_count = total_rows - missing_count

            missing_rate = missing_count / total_rows if total_rows > 0 else 0.0

            profile[column] = ColumnMissingRateProfile(
                column_name=column,
                missing_count=missing_count,
                non_missing_count=non_missing_count,
                total_count=total_rows,
                missing_rate=missing_rate
            )

        return profile

    def profile_rows(self, df: pl.DataFrame, sample_size: int) -> RowsMissingRateProfile:
        rows_missing_rates = self._row_missing_rates(df)
        full_missing_rows_profile = self.profile_full_missing_rows(rows_missing_rates, sample_size)
        high_missing_rate_rows_profile = self.profile_high_missing_rate_rows(rows_missing_rates, threshold=0.8, sample_size=sample_size)

        total_rows = df.height

        return RowsMissingRateProfile(
            total_rows=total_rows,
            full_missing_rows_count=full_missing_rows_profile["count"],
            full_missing_rows_rate=full_missing_rows_profile["count"] / total_rows if total_rows > 0 else 0.0,
            full_missing_rows_sample_indices=full_missing_rows_profile["sample_indices"],
            high_missing_rate_rows_count=high_missing_rate_rows_profile["count"],
            high_missing_rate_rows_rate=high_missing_rate_rows_profile["count"] / total_rows if total_rows > 0 else 0.0,
            high_missing_rate_rows_sample_indices=high_missing_rate_rows_profile["sample_indices"],
            sample_size=sample_size
        )

    def profile_rows_distribution(self, df: pl.DataFrame) -> MissingnessDistributionProfile:

        rows_missing_rates = self._row_missing_rates(df)

        return self._distribution_from_series(rows_missing_rates)

    def profile_columns_distribution(self, df: pl.DataFrame) -> MissingnessDistributionProfile:

        total_rows = df.height
        null_counts = df.null_count()  # 1-row DataFrame
        # Per-column missing rate as a flat polars Series
        columns_missing_rates = pl.Series(
            [null_counts[col][0] / total_rows if total_rows > 0 else 0.0 for col in df.columns]
        )

        return self._distribution_from_series(columns_missing_rates)

    def profile_full_missing_rows(self, rows_missing_rates: pl.Series, sample_size: int) -> Dict[str, Any]:

        fully_missing_mask = rows_missing_rates == 1.0
        fully_missing_indices = np.flatnonzero(fully_missing_mask.to_numpy())
        fully_missing_count = len(fully_missing_indices)

        sample_indices = self._sample_indices(fully_missing_indices, sample_size)

        return {
            "count": fully_missing_count,
            "sample_indices": sample_indices
        }

    def profile_high_missing_rate_rows(self, rows_missing_rates: pl.Series, threshold: float, sample_size: int) -> Dict[str, Any]:

        high_missing_mask = rows_missing_rates >= threshold
        high_missing_indices = np.flatnonzero(high_missing_mask.to_numpy())
        high_missing_count = len(high_missing_indices)

        sample_indices = self._sample_indices(high_missing_indices, sample_size)

        return {
            "count": high_missing_count,
            "sample_indices": sample_indices
        }

    def _row_missing_rates(self, df: pl.DataFrame) -> pl.Series:
        """Per-row fraction of null values, equivalent to df.isna().mean(axis=1) in pandas."""
        n_cols = df.width
        if n_cols == 0:
            return pl.Series([0.0] * df.height)

        null_count_per_row = df.select(
            pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int64) for c in df.columns]).alias("null_count")
        )["null_count"]

        return (null_count_per_row / n_cols)

    
    def _distribution_from_series(self, rates: pl.Series) -> MissingnessDistributionProfile:
        return MissingnessDistributionProfile(
            mean_missing_rate=self._as_float(rates.mean()),
            median_missing_rate=self._as_float(rates.median()),
            std_missing_rate=self._as_float(rates.std()) if rates.len() > 1 else 0.0,
            min_missing_rate=self._as_float(rates.min()),
            max_missing_rate=self._as_float(rates.max()),
            p90_missing_rate=self._as_float(rates.quantile(0.90)),
            p95_missing_rate=self._as_float(rates.quantile(0.95)),
            p99_missing_rate=self._as_float(rates.quantile(0.99)),
        )
    
    @staticmethod
    def _as_float(value: object) -> float:
        """
        Narrow polars' stats return types (which can be float, int,
        Decimal, timedelta, or None depending on stub overloads) to a
        plain float for our (numeric, non-temporal) missing-rate series.
        None only occurs on empty/all-null series; 0.0 is a safe default.
        """
        if value is None:
            return 0.0
        return float(value)  # type: ignore[arg-type]

    def _sample_indices(self, indices: np.ndarray, sample_size: int) -> list:
        """Reproducibly sample row indices, mirroring pandas' .sample(random_state=42)."""
        n = min(len(indices), sample_size)
        if n == 0:
            return []
        rng = np.random.RandomState(42)
        sampled = rng.choice(indices, size=n, replace=False)
        return sampled.tolist()