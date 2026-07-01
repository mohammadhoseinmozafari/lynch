from pydantic import BaseModel, UUID4, Field
from typing import Optional
from case.domain.value_objects.code_snapshot import CodeSnapshot
from case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from case.domain.value_objects.dataset_binding import DatasetBinding
from case.domain.value_objects.hyperparameter_bundle import HyperparameterBundle
from case.domain.value_objects.metric_bundle import MetricBundle
from case.domain.value_objects.model_artifact import ModelArtifactPointer
from case.application.dtos.register_model_request import RegisterModelRequest
class RegisterModelRequestSchema(BaseModel) :
    project_id : UUID4
    family_id : UUID4
    persona : Optional[str] = "Generic"
    code_snapshot: CodeSnapshot
    environment_snapshot: EnvironmentSnapshot
    dataset_binding: DatasetBinding
    hyperparameter_bundle : HyperparameterBundle
    metric_bundle : MetricBundle
    model_artifact : ModelArtifactPointer
    parent_version_id : Optional[UUID4] = None
    indempotency_key : Optional[UUID4] = Field(default=None,  max_length=255) # Post MVP
    allow_uninterpretable : Optional[bool] = False
    
    def to_dto(self) -> RegisterModelRequest:

        return RegisterModelRequest(
            project_id=self.project_id,
            family_id=self.family_id,
            persona=self.persona,
            code_snapshot=self.code_snapshot,
            environment_snapshot=self.environment_snapshot,
            dataset_binding=self.dataset_binding,
            hyperparameter_bundle=self.hyperparameter_bundle,
            metric_bundle=self.metric_bundle,
            model_artifact=self.model_artifact,
            parent_version_id=self.parent_version_id,
            indempotency_key=self.indempotency_key,
            allow_uninterpretable=self.allow_uninterpretable

        )
