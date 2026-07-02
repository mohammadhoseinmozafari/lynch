import math

import numpy as np
import pandas as pd
import pytest

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from case.infrastructure.persistence.repositories.in_memory import (
    InMemoryDataArtifactRepository,
)
from core.context import InvestigationContext
from data_investigation.missingness.profilers.rate_profiler import (
    ColumnDistributionMissingRateProfiler,
    ColumnMissingRateProfiler,
    RowsDistributionMissingRateProfiler,
    RowsMissingRateProfiler,
)


def base_profile() -> DataProfile:
    profile = DataProfile(feature_profiles={})
    profile.add_capability("profile.base")
    return profile


def context_for(dataframe: pd.DataFrame, profile: DataProfile) -> InvestigationContext:
    return InvestigationContext.from_dataframe(
        dataframe,
        profile=profile,
        artifact_store=InMemoryDataArtifactRepository(),
    )


def test_column_profiler_returns_namespace_without_mutating_profile():
    profile = base_profile()

    dataframe = pd.DataFrame({"a": [1, None], "b": [None, None]})
    namespace = ColumnMissingRateProfiler().profile(context_for(dataframe, profile))

    assert namespace.name == "missingness.column_rates"
    assert namespace.metrics == {
        "missing_rate_by_column": {"a": 0.5, "b": 1.0},
        "missing_count_by_column": {"a": 1, "b": 2},
        "non_missing_count_by_column": {"a": 1, "b": 0},
        "total_rows": 2,
    }
    assert "missingness.column_rates" not in profile.capabilities
    assert profile.namespaces == {}


def test_rows_profiler_returns_deterministic_metrics_without_mutation():
    profile = base_profile()
    dataframe = pd.DataFrame(
            {
                "a": [None, None, None, 1],
                "b": [None, None, 2, 2],
            }
    )
    namespace = RowsMissingRateProfiler(sample_size=1, threshold=0.5).profile(
        context_for(dataframe, profile)
    )

    metrics = namespace.metrics
    assert namespace.name == "missingness.row_rates"
    assert metrics["full_missing_rows_count"] == 2
    assert metrics["full_missing_rows_rate"] == 0.5
    assert metrics["high_missing_rows_count"] == 1
    assert metrics["high_missing_rows_rate"] == 0.25
    assert metrics["sample_indices"] == {
        "full_missing_rows": [1],
        "high_missing_rows": [2],
    }
    assert profile.namespaces == {}


def test_distribution_reuses_existing_column_namespace_without_mutation():
    profile = base_profile()
    cached_namespace = ProfileNamespace(
        name="missingness.column_rates",
        metrics={"missing_rate_by_column": {"a": 0.0, "b": 1.0}},
    )
    profile.set_namespace(cached_namespace)

    dataframe = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
    namespace = ColumnDistributionMissingRateProfiler().profile(
        context_for(dataframe, profile)
    )

    assert namespace.metrics["mean"] == 0.5
    assert namespace.metrics["median"] == 0.5
    assert namespace.metrics["min"] == 0.0
    assert namespace.metrics["max"] == 1.0
    assert set(profile.namespaces) == {"missingness.column_rates"}


def test_rows_distribution_uses_row_rates():
    dataframe = pd.DataFrame({"a": [None, 1], "b": [None, 1]})
    namespace = RowsDistributionMissingRateProfiler().profile(
        context_for(dataframe, base_profile())
    )

    assert namespace.metrics["mean"] == 0.5
    assert namespace.metrics["median"] == 0.5
    assert namespace.metrics["min"] == 0.0
    assert namespace.metrics["max"] == 1.0
    assert not math.isnan(namespace.metrics["std"])


@pytest.mark.parametrize("sample_size", [-1, 1.5, True, "2"])
def test_rows_profiler_rejects_invalid_sample_size(sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        RowsMissingRateProfiler(sample_size=sample_size)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_rows_profiler_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        RowsMissingRateProfiler(threshold=threshold)


@pytest.mark.parametrize(
    "profiler", [RowsMissingRateProfiler(), RowsDistributionMissingRateProfiler()]
)
def test_row_profilers_reject_dataframes_without_columns(profiler):
    with pytest.raises(ValueError, match="at least one column"):
        dataframe = pd.DataFrame(index=[0])
        profiler.profile(context_for(dataframe, base_profile()))


def test_column_profiler_rejects_duplicate_columns():
    df = pd.DataFrame([[1, None]], columns=["value", "value"])

    with pytest.raises(ValueError, match="columns must be unique"):
        ColumnMissingRateProfiler().profile(context_for(df, base_profile()))


@pytest.mark.parametrize(
    "rates",
    [
        [0.5],
        {"a": -0.1},
        {"a": 1.1},
        {"a": np.nan},
        {"a": np.inf},
        {"a": True},
        {"wrong_column": 0.5},
    ],
)
def test_column_distribution_rejects_invalid_rate_metrics(rates):
    profile = base_profile()
    profile.set_namespace(
        ProfileNamespace(
            name="missingness.column_rates",
            metrics={"missing_rate_by_column": rates},
        )
    )

    with pytest.raises(ValueError):
        dataframe = pd.DataFrame({"a": [None, 1]})
        ColumnDistributionMissingRateProfiler().profile(context_for(dataframe, profile))


def test_column_distribution_rejects_rate_count_inconsistency():
    profile = base_profile()
    profile.set_namespace(
        ProfileNamespace(
            name="missingness.column_rates",
            metrics={
                "missing_rate_by_column": {"a": 0.5},
                "missing_count_by_column": {"a": 2},
                "total_rows": 2,
            },
        )
    )

    with pytest.raises(ValueError, match="inconsistent with its count"):
        dataframe = pd.DataFrame({"a": [None, 1]})
        ColumnDistributionMissingRateProfiler().profile(context_for(dataframe, profile))
