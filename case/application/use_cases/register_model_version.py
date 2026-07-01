
from case.application.dtos.register_model_request import RegisterModelRequest
from case.domain.services.lineage_engine import LineageEngine

class RegisterModelVersionUseCase:
    def __init__(self, 
                    validator_chain, 
                    lineage_engine: LineageEngine,
                    artifact_manager,
                    model_version_repo,
                    training_run_repo,
                    explanation_link_repo,
                    outbox_repo) -> None:
        self.validator_chain = validator_chain
        self.lineage_engine = lineage_engine
        self.artifact_manager = artifact_manager
        self.model_version_repo = model_version_repo
        self.training_run_repo = training_run_repo
        self.explanation_link_repo = explanation_link_repo
        self.outbox_repo = outbox_repo
        

    def execute(self, request: RegisterModelRequest)-> RegisterModelResult:
        # Idempotency Check for Post MVP comes here
        code_hash = self.lineage_engine.compute_code_hash(request.code_snapshot)
        env_hash = self.lineage_engine.compute_env_hash(request.environment_snapshot)
        dataset_hash= self.lineage_engine.compute_dataset_hash(request.dataset_binding)

        training_hash = self.lineage_engine.compute_training_run_hash(
            code_hash=code_hash,
            env_hash=env_hash,
            dataset_hash=dataset_hash,
            hyperparams=request.hyperparameter_bundle,
            metrics=request.metric_bundle
        )
