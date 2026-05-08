from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from artifact import ModelArtifactPointer
from dataset import DatasetBinding
from ..code import CodeSnapshot
from environment import EnvironmentSnapshot
from metric import MetricBundle
from hyperparameter import HyperparameterBundle
from personas import Persona

@dataclass
class RegisterModelRequest :
    project_id : UUID
    family_id : UUID
    persona : Persona
    code_snapshot: CodeSnapshot
    environment_snapshot: EnvironmentSnapshot
    dataset_binding: DatasetBinding
    hyperparameter_bundle : HyperparameterBundle
    metric_bundle : MetricBundle
    model_artifact : ModelArtifactPointer
    parent_version_id : Optional[UUID] = None
    indempotency_key : Optional[UUID] = None
    allow_uninterpretable : Optional[bool] = False
