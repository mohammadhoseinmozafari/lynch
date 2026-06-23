from dataclasses import dataclass


from core.observation.observation import ObservationBatch

@dataclass
class ExecutionContext :
    id :str
    batch : ObservationBatch

@dataclass
class HealthExecutionContext (ExecutionContext):
    pass

@dataclass
class StructuralExecutionContext (ExecutionContext):
    pass