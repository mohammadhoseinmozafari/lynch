# holmes/runtime.py

from __future__ import annotations
from api.dataset import Dataset
from api.model import Model
from core.service import HolmesService
from core.analyzer import AnalyzerRegistry
from core.resource import Resource, ResourceRegistry

class HolmesRuntime:
    def __init__(self) -> None:
        
        self.resources = ResourceRegistry(
        )
        
        self.analyzers = AnalyzerRegistry()
        self.analyzers.initialize_runtime()
        
        self.service = HolmesService(
            resources=self.resources,
            analyzers=self.analyzers,
        )

    def register_dataset(
        self,
        df,
        *,
        name: str | None = None,
    ) -> Dataset:
        
        df = Resource.from_dataframe(df)
        resource = self.resources.create_dataset(
            value = df,
            name = name
        )

        return Dataset(
            id=resource.id,
            name=resource.name or f"dataset-{resource.id[:8]}",
            service=self.service,
        )

    def register_model(
        self,
        model,
        *,
        name: str | None = None,
    ) -> Model:
        
        model = Resource.from_model(model)
        resource = self.resources.create_model(
            value = model,
            name=name,
        )

        return Model(
            id=resource.id,
            name=resource.name or f"model-{resource.id[:8]}",
            service=self.service,
        )


_runtime: HolmesRuntime | None = None


def get_runtime() -> HolmesRuntime:
    global _runtime

    if _runtime is None:
        _runtime = HolmesRuntime()

    return _runtime