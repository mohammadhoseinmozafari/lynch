from __future__ import annotations
from pydantic import BaseModel, ConfigDict,Field
from typing import List, Optional, Dict
from enum import Enum
from .schema import DataSchemaSnapshot
from .profile import DataProfileSnapshot
from .split import SplitDefinition



class DatasetBinding(BaseModel) :
    schema_snapshot : DataSchemaSnapshot
    profile_snapshot : DataProfileSnapshot
    split_definition : SplitDefinition
    model_config = ConfigDict(arbitrary_types_allowed=True)


