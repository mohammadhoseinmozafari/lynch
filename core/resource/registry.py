

from typing import Dict

from .resource import Resource, ResourceType


class ResourceRegistry:
    def __init__(self) -> None:
        self._resources : Dict[str , Resource ] = {}

    def create_dataset(self, value : Resource, *, name=None) -> Resource:
        self.validate_dataset(value)
        self._resources[value.id] = value
        return value

    def create_model(self, value : Resource, *, name=None) -> Resource:
        self.validate_model (value)
        self._resources[value.id] = value
        return value

    def load(self, resource_id) -> Resource:
        return self._resources[resource_id]
    
    def validate_dataset(self , value : Resource)  -> None: 
        if not value.resource_type == ResourceType.DATASET:
            raise ValueError("Input value must be a dataset")

    def validate_model(self, value : Resource) -> None: 
        if not value.resource_type == ResourceType.MODEL:
            raise ValueError("Input value must be a model")
