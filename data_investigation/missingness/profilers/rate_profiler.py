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
    """
    Profiles missing-value statistics for every column in a dataset.

    This profiler computes column-level completeness metrics by counting
    missing and non-missing values for each column and calculating the
    corresponding missing rate.

    The profiler performs a single vectorized pass over the DataFrame,
    making it suitable for large datasets.

    Metrics produced for each column include:
        - Missing value count
        - Non-missing value count
        - Total row count
        - Missing rate

    Notes:
        - Missing values are detected using ``pandas.DataFrame.isna()``.
        - Empty DataFrames return a missing rate of ``0.0`` for all columns.
        - Runtime complexity is approximately O(rows × columns).

    Returns:
        Dict[str, ColumnMissingRateProfile]:
            Mapping from column name to its missing-value profile.
    """
    def __init__(self) -> None:
        super().__init__()

    def profile(self, df: pd.DataFrame) -> Dict[str, ColumnMissingRateProfile]:
        """
        Generate missing-value statistics for every column.

        Args:
            df:
                Input DataFrame to profile.

        Returns:
            Dictionary keyed by column name containing
            ``ColumnMissingRateProfile`` objects.

        Raises:
            None.
        """
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
    """
    Profiles row-level missing-value statistics for a dataset.

    This profiler identifies rows with excessive missing values and
    summarizes dataset quality from a row perspective.

    Two categories of problematic rows are reported:

        1. Fully missing rows
        2. Rows whose missing rate exceeds a configurable threshold

    Random samples of matching row indices are returned to help users
    inspect problematic records without scanning the entire dataset.

    Sampling is deterministic using a fixed random seed to ensure
    reproducible profiling results.

    Args:
        sample_size:
            Maximum number of example row indices returned for each
            category.

        threshold:
            Minimum fraction of missing values required for a row to be
            classified as highly incomplete.
    """
    def __init__(self, sample_size: int = 10, threshold: float = 0.8) -> None:
        super().__init__()
        self.id = str(ulid())
        self.profiler_type = ProfilerType.DATASET
        self.sample_size = sample_size
        self.threshold = threshold

    def profile(self, df: pd.DataFrame) -> RowsMissingRateProfile:
        """
        Profile row-level missingness.

        Computes:

        - Total dataset size
        - Number of fully missing rows
        - Rate of fully missing rows
        - Number of high-missing rows
        - Rate of high-missing rows
        - Sample row indices for both categories

        Args:
            df:
                Dataset to profile.

        Returns:
            A ``RowsMissingRateProfile`` summarizing row completeness.
        """
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
        """
        Identify rows containing only missing values.

        Args:
            rows_missing_rates:
                Per-row missing-value ratios.

            sample_size:
                Maximum number of row indices to sample.

        Returns:
            Dictionary containing:

            - ``count``: Total fully missing rows.
            - ``sample_indices``: Random sample of matching row indices.

        Notes:
            Sampling is deterministic using ``random_state=42``.
        """
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
        """
        Identify rows with high missing-value ratios.

        A row is considered highly incomplete when:

            threshold <= missing_rate < 1.0

        Fully missing rows are intentionally excluded because they are
        reported separately.

        Args:
            rows_missing_rates:
                Per-row missing-value ratios.

            threshold:
                Missing-rate threshold used for classification.

            sample_size:
                Maximum number of sampled row indices.

        Returns:
            Dictionary containing:

            - ``count``: Number of matching rows.
            - ``sample_indices``: Representative sampled indices.

        Notes:
            Sampling is deterministic using ``random_state=42``.
        """
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
    """
    Computes descriptive statistics for missing-value distributions.

    Depending on the configured axis, the profiler summarizes either:

        - Column missing rates
        - Row missing rates

    The resulting statistical profile provides a compact view of how
    missing values are distributed across the dataset.

    Statistics include:

        - Mean
        - Median
        - Standard deviation
        - Minimum
        - Maximum
        - 90th percentile
        - 95th percentile
        - 99th percentile

    Args:
        axis:
            Direction used when computing missing rates.

            - ``"columns"`` computes row-level missing rates.
            - ``"index"`` computes column-level missing rates.
    """
    def __init__(self, axis: Literal["index", "columns"] = "columns") -> None:
        super().__init__()
        self.axis = axis

    def profile(self, df: pd.DataFrame) -> MissingnessDistributionProfile:
        """
        Compute descriptive statistics for missing-value rates.

        Args:
            df:
                Dataset to profile.

        Returns:
            ``MissingnessDistributionProfile`` containing summary
            statistics describing the missing-rate distribution.
        """
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
    """
    High-level facade for dataset missing-value profiling.

    This class aggregates multiple specialized profilers and exposes a
    unified interface for analyzing missing values at different levels
    of granularity.

    Available analyses include:

        - Per-column missing statistics
        - Row-level missing statistics
        - Row missing-rate distribution
        - Column missing-rate distribution

    The class delegates work to dedicated profiler implementations while
    providing a simplified API for downstream consumers.

    Args:
        sample_size:
            Default number of sampled row indices returned by row-level
            profiling.

        threshold:
            Missing-rate threshold used when identifying highly
            incomplete rows.
    """
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
        """
        Profile missing values for every column.

        Args:
            df:
                Dataset to profile.

        Returns:
            Dictionary of column-level missing-value profiles.
        """
        return self._column_profiler.profile(df)

    def profile_rows(self, df: pd.DataFrame, sample_size: int) -> RowsMissingRateProfile:
        """
        Profile row-level missing values.

        Args:
            df:
                Dataset to profile.

            sample_size:
                Maximum number of sampled row indices to include in the
                result.

        Returns:
            Row-level missing-value summary.
        """
        return RowsMissingRateProfiler(
            sample_size=sample_size,
            threshold=self._rows_profiler.threshold,
        ).profile(df)

    def profile_distribution(self, df: pd.DataFrame) -> MissingnessDistributionProfile:
        """
        Compute the distribution of row missing rates.

        This method is equivalent to ``profile_rows_distribution()`` and
        is provided for convenience.

        Args:
            df:
                Dataset to profile.

        Returns:
            Statistical summary of row missing rates.
        """
        return self._rows_distribution_profiler.profile(df)

    def profile_rows_distribution(
        self,
        df: pd.DataFrame,
    ) -> MissingnessDistributionProfile:
        """
        Compute descriptive statistics for row missing rates.

        Args:
            df:
                Dataset to profile.

        Returns:
            Distribution summary of row completeness.
        """
        return self._rows_distribution_profiler.profile(df)

    def profile_columns_distribution(
        self,
        df: pd.DataFrame,
    ) -> MissingnessDistributionProfile:
        """
        Compute descriptive statistics for column missing rates.

        Args:
            df:
                Dataset to profile.

        Returns:
            Distribution summary of column completeness.
        """
        return self._columns_distribution_profiler.profile(df)

