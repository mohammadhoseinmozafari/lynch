import pandas as pd

from case.domain.enums.data_feature_type import DataFeatureType
from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from case.infrastructure.persistence.repositories.in_memory import (
    InMemoryDataArtifactRepository,
)
from core.context import InvestigationContext
from data_investigation.base.base_profiler import BaseDatasetProfiler


def empty_profile() -> DataProfile:
    return DataProfile(feature_profiles={})


def context_for(dataframe: pd.DataFrame, profile: DataProfile) -> InvestigationContext:
    return InvestigationContext.from_dataframe(
        dataframe,
        profile=profile,
        artifact_store=InMemoryDataArtifactRepository(),
    )


def test_base_profiler_returns_namespace_without_mutating_profile():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Boston", "New York", "Boston"],
            "active": [True, False, True],
        }
    )
    profile = empty_profile()

    namespace = BaseDatasetProfiler(dataset_name="users").profile(
        context_for(dataframe, profile)
    )

    assert isinstance(namespace, ProfileNamespace)
    assert namespace.name == "profile.base"
    assert set(namespace.metrics["feature_profiles"]) == {"age", "city", "active"}
    assert profile.feature_profiles == {}
    assert profile.capabilities == set()
    assert profile.namespaces == {}


def test_base_profiler_infers_feature_types():
    profile = empty_profile()
    dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "city": ["Boston", "New York", "Boston"],
            }
    )
    namespace = BaseDatasetProfiler().profile(context_for(dataframe, profile))

    feature_profiles = namespace.metrics["feature_profiles"]
    assert feature_profiles["age"].inferred_semantic_type == DataFeatureType.NUMERIC
    assert (
        feature_profiles["city"].inferred_semantic_type
        == DataFeatureType.CATEGORICAL
    )


def test_base_profiler_rejects_dataframes_without_columns():
    try:
        dataframe = pd.DataFrame()
        BaseDatasetProfiler().profile(context_for(dataframe, empty_profile()))
    except ValueError as error:
        assert str(error) == "df must contain at least one column"
    else:
        raise AssertionError("Expected a DataFrame without columns to be rejected")
