import pandas as pd

from case.domain.enums.data_feature_type import DataFeatureType
from case.domain.enums.data_type import DataType
from case.domain.value_objects.data_feature_schema import FeatureSchema
from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from case.domain.value_objects.dataset_artifact import InMemoryDataArtifactPointer
from case.domain.value_objects.dataset_binding import DatasetBinding
from case.domain.value_objects.data_schema import DataSchema
from core.profile import BaseDatasetProfiler, DatasetProfiler


def empty_profile() -> DataProfile:
    return DataProfile(feature_profiles={})


def test_base_profiler_enriches_supplied_profile():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Boston", "New York", "Boston"],
            "active": [True, False, True],
        }
    )
    profile = empty_profile()

    assert BaseDatasetProfiler(dataset_name="users").profile(dataframe, profile) is None

    assert profile.has_capability("profile.base")
    assert set(profile.feature_profiles) == {"age", "city", "active"}


def test_base_profiler_builds_feature_statistics():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Boston", "New York", "Boston"],
        }
    )
    profile = empty_profile()

    BaseDatasetProfiler().profile(dataframe, profile)

    age_profile = profile.feature_profiles["age"]
    city_profile = profile.feature_profiles["city"]
    assert age_profile.inferred_semantic_type == DataFeatureType.NUMERIC
    assert age_profile.feature_stats.mean == 30.0
    assert age_profile.feature_stats.range.min == 20.0
    assert age_profile.feature_stats.range.max == 40.0
    assert city_profile.inferred_semantic_type == DataFeatureType.CATEGORICAL
    assert city_profile.feature_stats.cardinality == 2
    assert city_profile.feature_stats.top_values["Boston"] == 2 / 3



def test_base_profiler_rejects_dataframes_without_columns():
    try:
        BaseDatasetProfiler().profile(pd.DataFrame(), empty_profile())
    except ValueError as error:
        assert str(error) == "df must contain at least one column"
    else:
        raise AssertionError("Expected a DataFrame without columns to be rejected")
