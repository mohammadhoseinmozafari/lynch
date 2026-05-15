from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional

from core.case.domain.enums.env_snapshot_type import EnvSnapshotType

class EnvironmentSnapshot (BaseModel) :
    snapshot_type : EnvSnapshotType
    specification : str
    python_version : str
    system_packages : List[str]
