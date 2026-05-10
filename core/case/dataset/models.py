from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from .schema import DataSchema
from .profile import DataProfile


class DatasetBinding(BaseModel) :
    schema_snapshot : DataSchema
    profile_snapshot : DataProfile
    model_config = ConfigDict(arbitrary_types_allowed=True)


