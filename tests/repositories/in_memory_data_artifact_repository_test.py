import pandas as pd
import pytest
from pydantic import ValidationError

from case.domain.repositories.data_artifact_repository import DataArtifactRepository
from case.domain.value_objects.dataset_artifact import (
    InMemoryDataArtifactPointer,
)
from case.infrastructure.persistence.repositories.in_memory import (
    DataArtifactNotFoundError,
    InMemoryDataArtifactRepository,
    UnsupportedDataArtifactPointerError,
)


def test_repository_contract_is_abstract():
    with pytest.raises(TypeError):
        DataArtifactRepository()


def test_store_returns_metadata_only_pointer_and_resolves_same_dataframe():
    repository = InMemoryDataArtifactRepository()
    dataframe = pd.DataFrame({"age": [20, 30], "city": ["Boston", "New York"]})

    pointer = repository.store(dataframe)

    assert isinstance(pointer, InMemoryDataArtifactPointer)
    assert pointer.object_id.startswith("memory://")
    assert pointer.record_count == 2
    assert pointer.column_count == 2
    assert not hasattr(pointer, "dataframe")
    assert not hasattr(pointer, "resolve")
    assert repository.get(pointer.object_id) is dataframe
    assert repository.exists(pointer.object_id) is True
    assert len(repository) == 1


def test_repository_preserves_by_reference_dataframe_behavior():
    repository = InMemoryDataArtifactRepository()
    dataframe = pd.DataFrame({"value": [1]})
    pointer = repository.store(dataframe)

    dataframe.loc[0, "value"] = 2

    assert repository.get(pointer.object_id).loc[0, "value"] == 2


def test_repository_instances_do_not_share_artifacts():
    owner = InMemoryDataArtifactRepository()
    other = InMemoryDataArtifactRepository()
    pointer = owner.store(pd.DataFrame({"value": [1]}))

    assert owner.exists(pointer.object_id) is True
    assert other.exists(pointer.object_id) is False
    with pytest.raises(DataArtifactNotFoundError):
        other.get(pointer.object_id)


def test_delete_removes_artifact_but_does_not_mutate_pointer():
    repository = InMemoryDataArtifactRepository()
    pointer = repository.store(pd.DataFrame({"value": [1, 2]}))

    repository.delete(pointer.object_id)

    assert repository.exists(pointer.object_id) is False
    assert pointer.record_count == 2
    assert len(repository) == 0
    with pytest.raises(DataArtifactNotFoundError):
        repository.get(pointer.object_id)
    with pytest.raises(DataArtifactNotFoundError):
        repository.delete(pointer.object_id)


def test_clear_removes_all_owned_artifacts():
    repository = InMemoryDataArtifactRepository()
    pointers = [
        repository.store(pd.DataFrame({"value": [index]}))
        for index in range(3)
    ]

    repository.clear()

    assert len(repository) == 0
    assert all(not repository.exists(pointer.object_id) for pointer in pointers)


def test_store_rejects_non_dataframe_artifacts():
    repository = InMemoryDataArtifactRepository()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        repository.store([{"value": 1}])


@pytest.mark.parametrize("method_name", ["get", "exists", "delete"])
def test_operations_reject_incompatible_object_ids(method_name: str):
    repository = InMemoryDataArtifactRepository()

    with pytest.raises(UnsupportedDataArtifactPointerError):
        getattr(repository, method_name)("s3://bucket/dataframe")


def test_pointer_is_immutable():
    pointer = InMemoryDataArtifactPointer(record_count=1, column_count=1)

    with pytest.raises(ValidationError):
        pointer.record_count = 2
