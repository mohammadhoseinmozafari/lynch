from types import SimpleNamespace

import pandas as pd
import pytest

from case.domain.value_objects.data_profile import DataProfile
from case.infrastructure.persistence.repositories.in_memory import (
    InMemoryDataArtifactRepository,
)
from core.context import InvestigationContext


def empty_profile() -> DataProfile:
    return DataProfile(feature_profiles={})


def test_from_dataframe_stores_and_resolves_dataframe():
    dataframe = pd.DataFrame({"value": [1, 2]})
    store = InMemoryDataArtifactRepository()

    context = InvestigationContext.from_dataframe(
        dataframe,
        profile=empty_profile(),
        artifact_store=store,
        metadata={"source": "test"},
    )

    assert context.get_dataframe() is dataframe
    assert context.artifact_pointer.record_count == 2
    assert context.artifact_pointer.column_count == 1
    assert context.dataset_binding is None
    assert context.metadata == {"source": "test"}


def test_runtime_parameters_can_be_read_and_updated():
    context = InvestigationContext.from_dataframe(
        pd.DataFrame({"value": [1]}),
        profile=empty_profile(),
        artifact_store=InMemoryDataArtifactRepository(),
        runtime_params={"threshold": 0.8},
    )

    assert context.get_param("threshold") == 0.8
    assert context.get_param("missing", 42) == 42

    context.set_param("threshold", 0.9)

    assert context.get_param("threshold") == 0.9


def test_runtime_parameter_defaults_are_not_shared():
    store = InMemoryDataArtifactRepository()
    first = InvestigationContext.from_dataframe(
        pd.DataFrame({"value": [1]}),
        profile=empty_profile(),
        artifact_store=store,
    )
    second = InvestigationContext.from_dataframe(
        pd.DataFrame({"value": [2]}),
        profile=empty_profile(),
        artifact_store=store,
    )

    first.set_param("sample_size", 10)

    assert second.get_param("sample_size") is None


def test_get_dataframe_requires_pointer_object_id():
    context = InvestigationContext(
        profile=empty_profile(),
        artifact_store=InMemoryDataArtifactRepository(),
        artifact_pointer=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="object_id"):
        context.get_dataframe()


def test_get_dataframe_rejects_non_dataframe_artifact():
    class InvalidStore:
        def get(self, object_id):
            return [object_id]

    context = InvestigationContext(
        profile=empty_profile(),
        artifact_store=InvalidStore(),
        artifact_pointer=SimpleNamespace(object_id="invalid://artifact"),
    )

    with pytest.raises(TypeError, match="pandas DataFrame"):
        context.get_dataframe()


def test_from_dataframe_requires_store_protocol():
    with pytest.raises(TypeError, match=r"store\(dataframe\)"):
        InvestigationContext.from_dataframe(
            pd.DataFrame({"value": [1]}),
            profile=empty_profile(),
            artifact_store=object(),
        )
