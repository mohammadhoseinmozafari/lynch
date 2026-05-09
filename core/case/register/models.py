from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from core.case.artifact import ModelArtifactPointer
from core.case.dataset import DatasetBinding
from core.case.code import CodeSnapshot
from core.case.environment import EnvironmentSnapshot
from core.case.metric import MetricBundle
from core.case.hyperparameter import HyperparameterBundle
from core.personas import Persona


class RegisterModelRequest(BaseModel) :
    project_id : UUID
    family_id : UUID
    persona : Optional[Persona]
    code_snapshot: CodeSnapshot
    environment_snapshot: EnvironmentSnapshot
    dataset_binding: DatasetBinding
    hyperparameter_bundle : HyperparameterBundle
    metric_bundle : MetricBundle
    model_artifact : ModelArtifactPointer
    parent_version_id : Optional[UUID] = None
    indempotency_key : Optional[UUID] = None
    allow_uninterpretable : Optional[bool] = False
