"""
Missingness Health Signal Processors
====================================

This module transforms missing-value profiling outputs into normalized
"health signals" that can be consumed by downstream systems such as
feature monitoring, dataset quality dashboards, or ML observability layers.

It operates on three levels of abstraction:

1. Column-level health signals
2. Row-level health signals
3. Dataset-level aggregated health signals

All processors convert raw missingness statistics into bounded metrics
within [0, 1], ensuring comparability across datasets.


"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from core.signal.enums import SubjectType
from core.signal.processor import SignalProcessor
from core.signal.signal import SignalTable
from data_investigation.missingness.profilers.models import (
    ColumnMissingRateProfile,
    MissingnessDistributionProfile,
    RowsMissingRateProfile,
)


def _profiles_frame(
    profiles: Mapping[str, ColumnMissingRateProfile],
) -> pd.DataFrame:
    """
    Convert a mapping of column profiles into a tabular representation.

    This function standardizes column-level missingness profiles into a
    DataFrame for vectorized signal computation.

    Args:
        profiles:
            Mapping of column name to ``ColumnMissingRateProfile``.

    Returns:
        DataFrame with schema:

            - column_name
            - missing_rate
            - missing_count
            - non_missing_count
            - total_count

        Returns an empty DataFrame with the above schema if input is empty.

    Notes:
        This is a pure transformation utility and does not validate
        semantic correctness of the input profiles.
    """
    if not profiles:
        return pd.DataFrame(
            columns=[
                "column_name",
                "missing_rate",
                "missing_count",
                "non_missing_count",
                "total_count",
            ]
        )

    return pd.DataFrame([profile.to_dict() for profile in profiles.values()])


def _single_profile_frame(profile: object) -> pd.DataFrame:
    """
    Convert a single profile object into a one-row DataFrame.

    This function standardizes dataset-level and row-level profiles into
    a consistent tabular format for downstream signal computation.

    Args:
        profile:
            Object exposing a ``to_dict()`` method.

    Returns:
        Single-row DataFrame representation of the profile.

    Raises:
        TypeError:
            If the input object does not implement ``to_dict()``.

    Notes:
        This enforces a minimal interface contract rather than a strict
        class hierarchy.
    """
    if hasattr(profile, "to_dict"):
        return pd.DataFrame([profile.to_dict()])
    raise TypeError("profile must provide a to_dict() method")


class ColumnarMissingnessHealthProcessor(
    SignalProcessor[Mapping[str, ColumnMissingRateProfile], SignalTable]
):
    """
    Column-level missingness health signal generator.

    This processor converts per-column missing-value statistics into
    bounded health signals suitable for feature quality monitoring.

    It evaluates four primary metrics per feature:

        - feature_completeness
        - feature_missingness
        - feature_usability
        - feature_observation_confidence

    These signals are designed for downstream ranking, alerting, and
    feature filtering pipelines.

    Configuration:
        config.acceptable_missing_rate:
            Maximum acceptable missing rate used to normalize usability.

    Output:
        SignalTable containing one row per feature with computed signals.

    Design Notes:
        - All outputs are clipped to [0, 1].
        - Confidence is modeled using a variance-aware estimator based on
          missing rate and sample size.
    """

    def __init__(self, config):
        self.config = config
        self.id = "columnar_missingness_health_signal_processor"
        self.subject_type = SubjectType.FEATURE
        self.supporting_type = []

    def run(
        self,
        profile: Mapping[str, ColumnMissingRateProfile],
    ) -> SignalTable:
        """
        Execute column-level health signal computation.

        Args:
            profile:
                Mapping of column names to missingness profiles.

        Returns:
            SignalTable with normalized feature-level health metrics.

        Behavior:
            - Empty input returns an empty SignalTable.
            - Missing rates are normalized against configured threshold.
            - Observation confidence decreases with uncertainty in
              sparsely observed columns.
        """
        threshold = float(self.config.acceptable_missing_rate)
        signal_df = _profiles_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        missing_rate = signal_df["missing_rate"].astype(float)
        total_count = signal_df["total_count"].astype(float)

        feature_observation_confidence = np.where(
            total_count > 0,
            1.0 - np.sqrt((missing_rate * (1.0 - missing_rate)) / total_count),
            0.0,
        )

        signal_df = signal_df.assign(
            feature_completeness=(1.0 - missing_rate).clip(0.0, 1.0),
            feature_missingness=missing_rate.clip(0.0, 1.0),
            feature_usability=(1.0 - (missing_rate / threshold)).clip(0.0, 1.0),
            feature_observation_confidence=np.clip(
                feature_observation_confidence,
                0.0,
                1.0,
            ),
        )

        return SignalTable(
            signal_df[
                [
                    "column_name",
                    "feature_completeness",
                    "feature_missingness",
                    "feature_usability",
                    "feature_observation_confidence",
                ]
            ]
        )


class RowsMissingnessHealthSignalProcessor(
    SignalProcessor[RowsMissingRateProfile, SignalTable]
):
    """
    Row-level missingness health signal generator.

    This processor evaluates dataset quality from a row integrity
    perspective by distinguishing between:

        - Completely missing rows
        - Partially missing rows

    It aggregates these into a single dataset-level integrity score.

    Configuration:
        config.full_missing_row_integrity_param:
            Weight applied to fully missing rows.

        config.high_missing_row_integrity_param:
            Weight applied to high-missing rows.

    Output:
        SignalTable with dataset-level row integrity metrics.
    """
    def __init__(self, config):
        self.config = config
        self.id = "rows_missingness_health_signal_processor"
        self.subject_type = SubjectType.ROW
        self.supporting_type = []

    def run(self, profile: RowsMissingRateProfile) -> SignalTable:
        """
        Compute row-level health signals.

        Args:
            profile:
                Row missingness profile containing dataset-level stats.

        Returns:
            SignalTable containing row integrity and failure rates.

        Notes:
            The integrity score is a weighted combination of:

                full_missing_rows_rate
                high_missing_rate_rows_rate

            All outputs are normalized to [0, 1].
        """
        signal_df = _single_profile_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        a = float(self.config.full_missing_row_integrity_param)
        b = float(self.config.high_missing_row_integrity_param)

        full_missing_rate = signal_df["full_missing_rows_rate"].astype(float)
        high_missing_rate = signal_df["high_missing_rate_rows_rate"].astype(float)

        signal_df = signal_df.assign(
            subject_name="dataset",
            dataset_row_integrity=(1.0 - (a * full_missing_rate + b * high_missing_rate)).clip(
                0.0,
                1.0,
            ),
            complete_row_failure_rate=full_missing_rate,
            partial_row_failure_rate=high_missing_rate,
            complete_row_health=(1.0 - full_missing_rate).clip(0.0, 1.0),
            partial_row_health=(1.0 - high_missing_rate).clip(0.0, 1.0),
        )

        return SignalTable(
            signal_df[
                [
                    "subject_name",
                    "dataset_row_integrity",
                    "complete_row_failure_rate",
                    "partial_row_failure_rate",
                    "complete_row_health",
                    "partial_row_health",
                ]
            ]
        )


class DatasetMissingnessHealthSignalProcessor(
    SignalProcessor[MissingnessDistributionProfile, SignalTable]
):
    """
    Dataset-level missingness health signal generator.

    This processor converts distributional missing-value statistics into
    high-level dataset quality signals.

    It summarizes both central tendency and worst-case behavior across
    features.

    Output metrics:

        - dataset_completeness
        - dataset_missingness
        - worst_feature_health
        - best_feature_health

    Configuration:
        No direct thresholds; operates purely on distribution statistics.

    Output:
        SignalTable with one dataset-level row.
    """
    def __init__(self, config):
        self.config = config
        self.id = "dataset_missingness_health_signal_processor"
        self.subject_type = SubjectType.DATASET
        self.supporting_type = []

    def run(self, profile: MissingnessDistributionProfile) -> SignalTable:
        """
        Compute dataset-level missingness health signals.

        Args:
            profile:
                Distributional missingness statistics across dataset.

        Returns:
            SignalTable containing dataset health summary.

        Notes:
            - Mean missing rate drives overall completeness.
            - Max/min missing rates define worst/best feature bounds.
            - All metrics are clipped to [0, 1].
        """
        signal_df = _single_profile_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        signal_df = signal_df.assign(
            subject_name="dataset",
            dataset_completeness=(1.0 - signal_df["mean_missing_rate"].astype(float)).clip(
                0.0,
                1.0,
            ),
            dataset_missingness=signal_df["mean_missing_rate"].astype(float).clip(0.0, 1.0),
            worst_feature_health=(1.0 - signal_df["max_missing_rate"].astype(float)).clip(
                0.0,
                1.0,
            ),
            best_feature_health=(1.0 - signal_df["min_missing_rate"].astype(float)).clip(
                0.0,
                1.0,
            ),
        )

        return SignalTable(
            signal_df[
                [
                    "subject_name",
                    "dataset_completeness",
                    "dataset_missingness",
                    "worst_feature_health",
                    "best_feature_health",
                ]
            ]
        )