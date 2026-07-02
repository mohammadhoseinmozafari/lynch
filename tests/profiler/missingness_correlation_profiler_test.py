import numpy as np
import pandas as pd
import pytest

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from case.infrastructure.persistence.repositories.in_memory import (
    InMemoryDataArtifactRepository,
)
from core.context import InvestigationContext
from data_investigation.missingness.profilers.correlation_profiler import (
    MissingnessCorrelationProfiler,
)


def profile_with_dependencies(df: pd.DataFrame) -> DataProfile:
    profile = DataProfile(feature_profiles={})
    profile.set_namespace(
        ProfileNamespace(name="profile.base", metrics={"total_rows": len(df)})
    )
    profile.set_namespace(
        ProfileNamespace(
            name="missingness.column_rates",
            metrics={
                "missing_count_by_column": {
                    column: int(df[column].isna().sum()) for column in df.columns
                }
            },
        )
    )
    return profile


def context_for(dataframe: pd.DataFrame, profile: DataProfile) -> InvestigationContext:
    return InvestigationContext.from_dataframe(
        dataframe,
        profile=profile,
        artifact_store=InMemoryDataArtifactRepository(),
    )


def test_correlated_missingness_has_high_jaccard_and_positive_phi():
    df = pd.DataFrame(
        {
            "left": [None, None, 1, 1, None, 1],
            "right": [None, None, 2, 2, None, 2],
        }
    )

    namespace = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    )
    pair = namespace.metrics["pairs"][0]

    assert pair["jaccard_similarity"] == 1.0
    assert pair["phi_correlation"] == pytest.approx(1.0)
    assert pair["valid_pair"] is True
    assert pair["invalid_reason"] is None


def test_independent_missingness_has_low_phi():
    df = pd.DataFrame(
        {
            "left": [None, None, 1, 1, None, None, 1, 1],
            "right": [None, 2, None, 2, None, 2, None, 2],
        }
    )

    pair = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    ).metrics["pairs"][0]

    assert pair["phi_correlation"] == pytest.approx(0.0)
    assert pair["jaccard_similarity"] == pytest.approx(1 / 3)


def test_pair_metrics_follow_binary_missingness_formulas():
    df = pd.DataFrame(
        {
            "left": [None, None, 1, 1],
            "right": [None, 2, None, None],
        }
    )

    pair = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    ).metrics["pairs"][0]

    assert pair["left_missing_count"] == 2
    assert pair["right_missing_count"] == 3
    assert pair["both_missing_count"] == 1
    assert pair["union_missing_count"] == 4
    assert pair["left_missing_rate"] == 0.5
    assert pair["right_missing_rate"] == 0.75
    assert pair["support"] == 0.25
    assert pair["jaccard_similarity"] == 0.25
    assert pair["p_right_missing_given_left_missing"] == 0.5
    assert pair["p_left_missing_given_right_missing"] == pytest.approx(1 / 3)
    assert pair["lift_left_to_right"] == pytest.approx(2 / 3)
    assert pair["lift_right_to_left"] == pytest.approx(2 / 3)
    assert pair["phi_correlation"] == pytest.approx(-1 / np.sqrt(3))


def test_fully_missing_column_has_none_phi_but_remains_valid():
    df = pd.DataFrame(
        {
            "always_missing": [None, None, None, None],
            "sometimes_missing": [None, 1, None, 1],
        }
    )

    pair = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    ).metrics["pairs"][0]

    assert pair["phi_correlation"] is None
    assert pair["valid_pair"] is True
    assert pair["invalid_reason"] is None


def test_namespace_shape_and_input_immutability():
    df = pd.DataFrame(
        {
            "a": [None, 1, None],
            "b": [None, None, 2],
            "c": [3, None, None],
        }
    )
    original_df = df.copy(deep=True)
    profile = profile_with_dependencies(df)
    original_namespace_names = set(profile.namespaces)

    namespace = MissingnessCorrelationProfiler().profile(context_for(df, profile))

    assert namespace.name == "missingness.correlation"
    assert namespace.metrics["total_rows"] == 3
    assert namespace.metrics["total_columns"] == 3
    assert namespace.metrics["eligible_column_count"] == 3
    assert namespace.metrics["pair_count"] == 3
    assert namespace.metrics["valid_pair_count"] == 3
    assert namespace.artifacts == {}
    assert namespace.metadata == {
        "profiler": "MissingnessCorrelationProfiler",
        "method": "pairwise_binary_missingness",
    }
    assert set(profile.namespaces) == original_namespace_names
    assert "missingness.correlation" not in profile.namespaces
    pd.testing.assert_frame_equal(df, original_df)


def test_pair_order_follows_original_column_order():
    df = pd.DataFrame(
        {
            "z": [None, 1],
            "a": [None, 1],
            "m": [None, 1],
            "b": [None, 1],
        }
    )

    pairs = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    ).metrics["pairs"]

    assert [(pair["left_column"], pair["right_column"]) for pair in pairs] == [
        ("z", "a"),
        ("z", "m"),
        ("z", "b"),
        ("a", "m"),
        ("a", "b"),
        ("m", "b"),
    ]


def test_matrix_product_does_not_overflow_uint8():
    df = pd.DataFrame(
        {
            "left": [np.nan] * 300,
            "right": [np.nan] * 300,
        }
    )

    pair = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    ).metrics["pairs"][0]

    assert pair["both_missing_count"] == 300
    assert pair["union_missing_count"] == 300


def test_pair_without_missing_values_is_invalid_and_zero_safe():
    df = pd.DataFrame({"left": [1, 2], "right": [3, 4]})

    pair = MissingnessCorrelationProfiler().profile(
        context_for(df, profile_with_dependencies(df))
    ).metrics["pairs"][0]

    assert pair["valid_pair"] is False
    assert pair["phi_correlation"] is None
    assert pair["jaccard_similarity"] == 0.0
    assert pair["lift_left_to_right"] == 0.0
    assert pair["lift_right_to_left"] == 0.0
    assert pair["invalid_reason"] == (
        "left_column_has_no_missing_values; "
        "right_column_has_no_missing_values"
    )


@pytest.mark.parametrize("bad_count", [True, -1, 3, 1.5, "1"])
def test_invalid_cached_missing_count_is_rejected(bad_count):
    df = pd.DataFrame({"left": [None, 1], "right": [None, 2]})
    profile = profile_with_dependencies(df)
    profile.namespaces["missingness.column_rates"].metrics[
        "missing_count_by_column"
    ]["left"] = bad_count

    with pytest.raises(ValueError, match="Invalid missing count for column 'left'"):
        MissingnessCorrelationProfiler().profile(context_for(df, profile))


def test_stale_cached_missing_count_is_rejected():
    df = pd.DataFrame({"left": [None, 1], "right": [None, 2]})
    profile = profile_with_dependencies(df)
    profile.namespaces["missingness.column_rates"].metrics[
        "missing_count_by_column"
    ]["left"] = 0

    with pytest.raises(ValueError, match="does not match DataFrame"):
        MissingnessCorrelationProfiler().profile(context_for(df, profile))


def test_cached_count_columns_must_match_dataframe():
    df = pd.DataFrame({"left": [None, 1], "right": [None, 2]})
    profile = profile_with_dependencies(df)
    profile.namespaces["missingness.column_rates"].metrics[
        "missing_count_by_column"
    ].pop("right")

    with pytest.raises(ValueError, match="columns do not match"):
        MissingnessCorrelationProfiler().profile(context_for(df, profile))


def test_cached_counts_must_be_a_mapping():
    df = pd.DataFrame({"left": [None, 1], "right": [None, 2]})
    profile = profile_with_dependencies(df)
    profile.namespaces["missingness.column_rates"].metrics[
        "missing_count_by_column"
    ] = [1, 1]

    with pytest.raises(ValueError, match="must be a mapping"):
        MissingnessCorrelationProfiler().profile(context_for(df, profile))


def test_correlation_profiler_rejects_duplicate_columns():
    df = pd.DataFrame([[None, None]], columns=["value", "value"])
    profile = DataProfile(feature_profiles={})
    profile.set_namespace(
        ProfileNamespace(name="profile.base", metrics={"total_rows": 1})
    )
    profile.set_namespace(
        ProfileNamespace(
            name="missingness.column_rates",
            metrics={"missing_count_by_column": {"value": 1}},
        )
    )

    with pytest.raises(ValueError, match="columns must be unique"):
        MissingnessCorrelationProfiler().profile(context_for(df, profile))


def test_impossible_contingency_table_is_rejected():
    with pytest.raises(ValueError, match="Both-missing count"):
        MissingnessCorrelationProfiler._build_pair(
            left_column="left",
            right_column="right",
            left_missing_count=1,
            right_missing_count=1,
            both_missing_count=2,
            total_rows=2,
        )


def test_missing_required_namespace_raises_domain_error():
    dataframe = pd.DataFrame({"left": [None], "right": [None]})
    with pytest.raises(KeyError):
        MissingnessCorrelationProfiler().profile(
            context_for(dataframe, DataProfile(feature_profiles={}))
        )


def test_missing_required_metric_raises_domain_error():
    profile = DataProfile(feature_profiles={})
    profile.set_namespace(ProfileNamespace(name="missingness.column_rates"))
    profile.set_namespace(
        ProfileNamespace(name="profile.base", metrics={"total_rows": 1})
    )

    dataframe = pd.DataFrame({"left": [None], "right": [None]})
    with pytest.raises(KeyError):
        MissingnessCorrelationProfiler().profile(
            context_for(dataframe, profile)
        )
