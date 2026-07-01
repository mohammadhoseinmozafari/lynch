
from abc import ABC, abstractmethod
from typing import Optional

from case.domain.value_objects import CodeSnapshot
from case.domain.value_objects.dataset_binding import DatasetBinding
from case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from case.domain.value_objects.hyperparameter_bundle import HyperparameterBundle
from case.domain.value_objects.metric_bundle import MetricBundle


class LineageEngine(ABC):

    @abstractmethod
    def compute_code_hash(self, code: CodeSnapshot) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def compute_dataset_hash(self, binding: DatasetBinding)-> str:
        raise NotImplementedError()
    
    @abstractmethod
    def compute_env_hash (self, env: EnvironmentSnapshot) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def compute_training_run_hash(self,
                                    code_hash: str,
                                    env_hash: str,
                                    dataset_hash: str,
                                    hyperparams: HyperparameterBundle,
                                    metrics: MetricBundle,
                                    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def compute_version_hash(self, 
                                    training_run_hash : str, 
                                    parent_version_hash: str | None
                                    )-> str:
        raise NotImplementedError()
        
