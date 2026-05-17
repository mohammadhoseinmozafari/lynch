
class RegisterModelVersionUseCase:
    def __init__(self, 
                    validator_chain, 
                    lineage_engine,
                    artifacte_manager,
                    model_version_repo,
                    training_run_repo,
                    explanation_link_repo,
                    outbox_repo) -> None:
        pass

    def execute(self, request: RegisterModelRequest)-> RegisterModelResult:
        pass