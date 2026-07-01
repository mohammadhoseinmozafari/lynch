from case.presentation.api.schemas.request_schemas import RegisterModelRequestSchema
from case.application.use_cases.register_model_version import RegisterModelVersionUseCase
class RegistrationController :

    def register_model_version (self, 
                       request : RegisterModelRequestSchema, use_case: RegisterModelVersionUseCase ) -> RegisterModelResponseSchema:
        
        try :
            result = use_case.execute(request=request.to_dto())
        except Exception as e:
            print(e)
        
