# holmes/api/model.py

from __future__ import annotations

from typing import Any
from .dataset import Dataset


class Model:
    def __init__(
        self,
        *,
        id: str,
        name: str,
        service: Any,
    ) -> None:
        self._id = id
        self._name = name
        self._service = service

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def perturb(
        self,
        dataset: Dataset,
        *,
        feature: str,
    ) -> Any:
        return self._service.perturb_model(
            model_id=self._id,
            dataset_id=dataset.id,
            feature=feature,
        )