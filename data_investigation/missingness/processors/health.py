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


class ColumnarMissingnessHealthProcessor(
    SignalProcessor[Mapping[str, ColumnMissingRateProfile], SignalTable]
):
    def __init__(self, config):
        self.config = config
        self.id = "columnar_missingness_health_signal_processor"
        self.subject_type = SubjectType.FEATURE
        self.supporting_type = []

    def run(
        self,
        profile: Mapping[str, ColumnMissingRateProfile],
    ) -> SignalTable:
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
    def __init__(self, config):
        self.config = config
        self.id = "rows_missingness_health_signal_processor"
        self.subject_type = SubjectType.ROW
        self.supporting_type = []

    def run(self, profile: RowsMissingRateProfile) -> SignalTable:
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
    def __init__(self, config):
        self.config = config
        self.id = "dataset_missingness_health_signal_processor"
        self.subject_type = SubjectType.DATASET
        self.supporting_type = []

    def run(self, profile: MissingnessDistributionProfile) -> SignalTable:
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