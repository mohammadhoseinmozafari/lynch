import math

import pandas as pd
import pytest

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from data_investigation.missingness.profilers.rate_profiler import (
    ColumnMissingRateProfiler,
    DistributionMissingRateProfiler,
    RowsMissingRateProfiler,
)


def base_profile() -> DataProfile:
    profile = DataProfile(feature_profiles={})
    profile.add_capability("profile.base")
    return profile



def test_rows_profiler_writes_deterministic_samples_and_metrics():
    dataframe = pd.DataFrame(
        {
            "a": [None, None, None, 1],
            "b": [None, None, 2, 2],
        }
    )
    profile = base_profile()

    RowsMissingRateProfiler(sample_size=1, threshold=0.5).profile(dataframe, profile)

    metrics = profile.namespaces["missingness.row_rates"].metrics
    assert metrics["full_missing_rows_count"] == 2
    assert metrics["full_missing_rows_rate"] == 0.5
    assert metrics["high_missing_rows_count"] == 1
    assert metrics["high_missing_rows_rate"] == 0.25
    assert metrics["sample_indices"] == {
        "full_missing_rows": [1],
        "high_missing_rows": [2],
    }
    assert metrics["threshold"] == 0.5
    assert metrics["sample_size"] == 1
    assert metrics["total_rows"] == 4


def test_distribution_reuses_column_namespace():
    profile = base_profile()
    profile.set_namespace(
        ProfileNamespace(
            name="missingness.column_rates",
            metrics={"missing_rate_by_column": {"a": 0.0, "b": 1.0}},
        )
    )
    profile.add_capability("missingness.column_rates")

    # The dataframe deliberately disagrees with the cached values.
    DistributionMissingRateProfiler().profile(
        pd.DataFrame({"a": [1, 1], "b": [1, 1]}), profile
    )

    metrics = profile.namespaces["missingness.distribution"].metrics
    assert metrics["mean"] == 0.5
    assert metrics["median"] == 0.5
    assert metrics["min"] == 0.0
    assert metrics["max"] == 1.0


def test_distribution_falls_back_to_row_rates():
    profile = base_profile()
    DistributionMissingRateProfiler().profile(
        pd.DataFrame({"a": [None, 1], "b": [None, 1]}), profile
    )

    metrics = profile.namespaces["missingness.distribution"].metrics
    assert metrics["mean"] == 0.5
    assert metrics["median"] == 0.5
    assert metrics["min"] == 0.0
    assert metrics["max"] == 1.0
    assert not math.isnan(metrics["std"])
