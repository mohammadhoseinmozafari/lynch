from __future__ import annotations
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class EnvironmentSnapshot (BaseModel) :
    snapshot_type : SnapshotType
    specification : str
    python_version : str
    system_packages : List[str]
    cuda_version : Optional[str] = None


class SnapshotType (Enum) :
    CONDA_LOCK = 'conda-lock'
    PIP_LOCK =  'pip-lock'
    DOCKER_DIGEST = 'docker-digest'
