import pandas as pd

from case.domain.enums.data_feature_type import DataFeatureType
from case.domain.enums.data_type import DataType
from case.domain.value_objects.dataset_artifact import InMemoryDataArtifactPointer
from core.profile import BaseDatasetProfiler, DatasetProfiler


def test_base_profiler_creates_binding_with_in_memory_dataframe_pointer():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Boston", "New York", "Boston"],
            "active": [True, False, True],
        }
    )

    binding = BaseDatasetProfiler(dataset_name="users").profile(dataframe)

    assert binding.dataset_name == "users"
    assert isinstance(binding.artifact_pointer, InMemoryDataArtifactPointer)
    assert binding.artifact_pointer.resolve() is dataframe
    assert binding.artifact_pointer.record_count == 3
    assert binding.artifact_pointer.column_count == 3
    assert binding.dataset_profile.has_capability("profile.base")


def test_base_profiler_builds_schema_and_feature_statistics():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Boston", "New York", "Boston"],
        }
    )

    binding = BaseDatasetProfiler().profile(dataframe)

    age_schema = binding.dataset_schema.features["age"]
    age_profile = binding.dataset_profile.feature_profiles["age"]
    city_schema = binding.dataset_schema.features["city"]
    city_profile = binding.dataset_profile.feature_profiles["city"]

    assert age_schema.dtype == DataType.INT64
    assert age_schema.feature_type == DataFeatureType.NUMERIC
    assert age_profile.feature_stats.mean == 30.0
    assert age_profile.feature_stats.range.min == 20.0
    assert age_profile.feature_stats.range.max == 40.0
    assert city_schema.feature_type == DataFeatureType.CATEGORICAL
    assert city_profile.feature_stats.cardinality == 2
    assert city_profile.feature_stats.top_values["Boston"] == 2 / 3


def test_profile_binding_adds_each_profiler_capability_after_success():
    class RowCountProfiler(DatasetProfiler):
        capability = "profile.row_count"

        def profile(self, df: pd.DataFrame) -> int:
            return len(df)

    binding = BaseDatasetProfiler().profile(pd.DataFrame({"value": [1, 2, 3]}))

    result = RowCountProfiler().profile_binding(binding)

    assert result == 3
    assert binding.dataset_profile.capabilities == {
        "profile.base",
        "profile.row_count",
    }


def test_profile_binding_derives_capability_when_profiler_does_not_declare_one():
    class DistributionProfiler(DatasetProfiler):
        def profile(self, df: pd.DataFrame) -> int:
            return len(df.columns)

    binding = BaseDatasetProfiler().profile(pd.DataFrame({"value": [1]}))

    DistributionProfiler().profile_binding(binding)

    assert binding.dataset_profile.has_capability("profile.distribution")


def test_base_profiler_rejects_dataframes_without_columns():
    dataframe = pd.DataFrame()

    try:
        BaseDatasetProfiler().profile(dataframe)
    except ValueError as error:
        assert str(error) == "df must contain at least one column"
    else:
        raise AssertionError("Expected a DataFrame without columns to be rejected")
