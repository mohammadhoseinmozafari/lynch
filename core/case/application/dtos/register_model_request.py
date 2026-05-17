
from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from core.case.domain.value_objects.code_snapshot import CodeSnapshot
from core.case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from core.case.domain.value_objects.dataset_binding import DatasetBinding
from core.case.domain.value_objects.hyperparameter_bundle import HyperparameterBundle
from core.case.domain.value_objects.metric_bundle import MetricBundle
from core.case.domain.value_objects.model_artifact import ModelArtifactPointer

@dataclass
class RegisterModelRequest :
    project_id : UUID
    family_id : UUID
    persona : Optional[str] 
    code_snapshot: CodeSnapshot
    environment_snapshot: EnvironmentSnapshot
    dataset_binding: DatasetBinding
    hyperparameter_bundle : HyperparameterBundle
    metric_bundle : MetricBundle
    model_artifact : ModelArtifactPointer
    parent_version_id : Optional[UUID] 
    indempotency_key : Optional[UUID]  # Post MVP
    allow_uninterpretable : Optional[bool] 
    