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
    if hasattr(profile, "to_dict"):
        return pd.DataFrame([profile.to_dict()])
    raise TypeError("profile must provide a to_dict() method")


class ColumnarMissingnessStructuralSignalProcessor(
    SignalProcessor[Mapping[str, ColumnMissingRateProfile], SignalTable]
):
    def __init__(self, config):
        self.config = config
        self.id = "columnar_missingness_structural_signal_processor"
        self.supporting_type = []
        self.subject_type = SubjectType.FEATURE

    def run(
        self,
        profile: Mapping[str, ColumnMissingRateProfile],
    ) -> SignalTable:
        signal_df = _profiles_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        z_thresh = float(getattr(self.config, "localized_feature_failure_z_threshold", 2.0))
        global_std_thresh = float(getattr(self.config, "global_degradation_std_threshold", 0.02))
        global_mean_thresh = float(getattr(self.config, "global_degradation_mean_threshold", 0.1))
        dominant_thresh = float(getattr(self.config, "dominant_feature_failure_threshold", 0.5))

        signal_df = signal_df.sort_values("column_name").reset_index(drop=True)
        missing_rate = signal_df["missing_rate"].astype(float)
        missing_count = signal_df["missing_count"].astype(float)

        mean_missing = float(missing_rate.mean())
        std_missing = float(missing_rate.std(ddof=0))
        std_for_z = std_missing if std_missing > 0 else 1e-9
        total_missing = float(missing_count.sum())

        q1 = float(missing_rate.quantile(0.25))
        q3 = float(missing_rate.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        dominant_ratio = np.where(total_missing > 0, missing_count / total_missing, 0.0)

        signal_df = signal_df.assign(
            localized_feature_failure=((missing_rate - mean_missing) / std_for_z > z_thresh).astype(int),
            global_feature_degradation=int(std_missing < global_std_thresh and mean_missing > global_mean_thresh),
            dominant_feature_failure_ratio=dominant_ratio,
            dominant_feature_failure=(dominant_ratio > dominant_thresh).astype(int),
            feature_missingness_outlier=((missing_rate < lower) | (missing_rate > upper)).astype(int),
        )

        return SignalTable(
            signal_df[
                [
                    "column_name",
                    "localized_feature_failure",
                    "global_feature_degradation",
                    "dominant_feature_failure_ratio",
                    "dominant_feature_failure",
                    "feature_missingness_outlier",
                ]
            ]
        )


class RowsMissingnessStructuralSignalProcessor(
    SignalProcessor[RowsMissingRateProfile, SignalTable]
):
    def __init__(self, config):
        self.config = config
        self.id = "rows_missingness_structural_signal_processor"
        self.supporting_type = []
        self.subject_type = SubjectType.ROW

    def run(self, profile: RowsMissingRateProfile) -> SignalTable:
        signal_df = _single_profile_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        complete_thresh = float(getattr(self.config, "complete_row_corruption_threshold", 0.1))
        partial_thresh = float(getattr(self.config, "partial_row_corruption_threshold", 0.2))
        mixed_full_thresh = float(getattr(self.config, "mixed_row_corruption_full_threshold", 0.1))
        mixed_high_thresh = float(getattr(self.config, "mixed_row_corruption_high_threshold", 0.2))

        signal_df = signal_df.assign(
            subject_name="dataset",
            complete_row_corruption=(signal_df["full_missing_rows_rate"].astype(float) > complete_thresh).astype(int),
            partial_row_corruption=(signal_df["high_missing_rate_rows_rate"].astype(float) > partial_thresh).astype(int),
            mixed_row_corruption=(
                (
                    signal_df["full_missing_rows_rate"].astype(float) > mixed_full_thresh
                )
                & (
                    signal_df["high_missing_rate_rows_rate"].astype(float) > mixed_high_thresh
                )
            ).astype(int),
        )

        return SignalTable(
            signal_df[
                [
                    "subject_name",
                    "complete_row_corruption",
                    "partial_row_corruption",
                    "mixed_row_corruption",
                ]
            ]
        )


class RowsDistributionStructuralSignalProcessor(
    SignalProcessor[MissingnessDistributionProfile, SignalTable]
):
    def __init__(self, config):
        self.config = config
        self.id = "rows_distribution_structural_signal_processor"
        self.supporting_type = []
        self.subject_type = SubjectType.ROW

    def run(self, profile: MissingnessDistributionProfile) -> SignalTable:
        signal_df = _single_profile_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        std_low = float(getattr(self.config, "uniform_row_std_threshold", 0.05))
        std_high = float(getattr(self.config, "heterogeneous_row_std_threshold", 0.1))
        heavy_tail_thresh = float(getattr(self.config, "heavy_tail_row_threshold", 0.3))
        skewed_thresh = float(getattr(self.config, "skewed_row_threshold", 0.1))
        extreme_max_thresh = float(getattr(self.config, "extreme_row_failure_max_threshold", 0.8))

        signal_df = signal_df.assign(
            subject_name="dataset",
            uniform_row_quality=(signal_df["std_missing_rate"].astype(float) < std_low).astype(int),
            heterogeneous_row_quality=(signal_df["std_missing_rate"].astype(float) > std_high).astype(int),
            heavy_tail_row_corruption=((signal_df["p99_missing_rate"].astype(float) - signal_df["median_missing_rate"].astype(float)) > heavy_tail_thresh).astype(int),
            skewed_row_corruption=((signal_df["mean_missing_rate"].astype(float) - signal_df["median_missing_rate"].astype(float)).abs() > skewed_thresh).astype(int),
            extreme_row_failure=(signal_df["max_missing_rate"].astype(float) > extreme_max_thresh).astype(int),
        )

        return SignalTable(
            signal_df[
                [
                    "subject_name",
                    "uniform_row_quality",
                    "heterogeneous_row_quality",
                    "heavy_tail_row_corruption",
                    "skewed_row_corruption",
                    "extreme_row_failure",
                ]
            ]
        )


class ColumnsDistributionStructuralSignalProcessor(
    SignalProcessor[MissingnessDistributionProfile, SignalTable]
):
    def __init__(self, config):
        self.config = config
        self.id = "columns_distribution_structural_signal_processor"
        self.supporting_type = []
        self.subject_type = SubjectType.FEATURE

    def run(self, profile: MissingnessDistributionProfile) -> SignalTable:
        signal_df = _single_profile_frame(profile)

        if signal_df.empty:
            return SignalTable(signal_df)

        std_low = float(getattr(self.config, "uniform_feature_std_threshold", 0.05))
        std_high = float(getattr(self.config, "heterogeneous_feature_std_threshold", 0.1))
        heavy_tail_thresh = float(getattr(self.config, "heavy_tail_feature_threshold", 0.3))
        skewed_thresh = float(getattr(self.config, "skewed_feature_threshold", 0.1))
        extreme_max_thresh = float(getattr(self.config, "extreme_feature_failure_max_threshold", 0.8))

        signal_df = signal_df.assign(
            subject_name="dataset",
            uniform_feature_quality=(signal_df["std_missing_rate"].astype(float) < std_low).astype(int),
            heterogeneous_feature_quality=(signal_df["std_missing_rate"].astype(float) > std_high).astype(int),
            heavy_tail_feature_corruption=((signal_df["p99_missing_rate"].astype(float) - signal_df["median_missing_rate"].astype(float)) > heavy_tail_thresh).astype(int),
            skewed_feature_corruption=((signal_df["mean_missing_rate"].astype(float) - signal_df["median_missing_rate"].astype(float)).abs() > skewed_thresh).astype(int),
            extreme_feature_failure=(signal_df["max_missing_rate"].astype(float) > extreme_max_thresh).astype(int),
        )

        return SignalTable(
            signal_df[
                [
                    "subject_name",
                    "uniform_feature_quality",
                    "heterogeneous_feature_quality",
                    "heavy_tail_feature_corruption",
                    "skewed_feature_corruption",
                    "extreme_feature_failure",
                ]
            ]
        )