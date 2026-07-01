# src/infrastructure/services/lineage_engine_impl.py
import hashlib
import json
from typing import Any
from case.domain.value_objects.code_snapshot import CodeSnapshot
from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.dataset_binding import DatasetBinding
from case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from case.domain.value_objects.hyperparameter_bundle import HyperparameterBundle
from case.domain.value_objects.metric_bundle import MetricBundle
from case.domain.services.lineage_engine import LineageEngine
from case.domain.services.canonical_serializer import CanonicalSerializer

class LineageEngineImpl(LineageEngine):
    def __init__(self, serializer: CanonicalSerializer):
        self._serializer = serializer

    def compute_code_hash(self, code: CodeSnapshot) -> str:
        # Include all fields that affect execution
        data = {
            "git_commit_hash": code.git_commit_hash,
            "entry_point": code.entry_point,
            "git_branch": code.git_branch,
            "training_command": code.training_command,
        }
        if code.uncommitted_diff:
            # Hash the diff itself to avoid huge strings in concatenation
            diff_hash = hashlib.sha256(code.uncommitted_diff.encode()).hexdigest()
            data["uncommitted_diff_hash"] = diff_hash
       
        json_str = self._serializer.to_json(data)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def compute_environment_hash(self, env: EnvironmentSnapshot) -> str:
        # Hash the normalized environment spec
        system_packages = env.system_packages or []
        data = {
            "snapshot_type": env.snapshot_type.value,
            "specification": env.specification,  # could be conda-lock.yml content
            "python_version": env.python_version,
            "system_packages": sorted(env.system_packages),
        }
        
        json_str = self._serializer.to_json(data)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def compute_dataset_hash(self, binding: DatasetBinding) -> str:
        # Include schema, profile (if any), row counts, and split URIs
        splits_info = {}
        for split_name, split in binding.data_splits.items():
            split_dict = {
                "artifact": {
                    "storage_backend": split.split_artifact.storage_backend,
                    "bucket": split.split_artifact.bucket,
                    "key": split.split_artifact.key,
                },
                "row_count": split.row_count,
                "split_fraction": split.split_fraction,
                "schema": self._serializer.to_json(split.data_schema),
            }
            if split.data_profile:
                split_dict["profile_hash"] = self._compute_profile_hash(split.data_profile)
            splits_info[split_name.value] = split_dict
        data = {
            "dataset_name": binding.dataset_name,
            "drift_baseline": binding.drift_baseline.value if binding.drift_baseline else None,
            "task_type": binding.task_type.value if binding.task_type else None,
            "splits": splits_info,
        }
        json_str = self._serializer.to_json(data)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_profile_hash(self, profile: DataProfile) -> str:
        # Hash the statistical profile (used for drift detection)
        data = {
            "computed_at": profile.computed_at.isoformat() if profile.computed_at else None,
            "features": {
                name: {
                    "missing_rate": fp.missing_rate,
                    "stats": self._serializer.to_json(fp.feature_stats),
                }
                for name, fp in profile.feature_profiles.items()
            },
        }
        json_str = self._serializer.to_json(data)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def compute_training_run_hash(
        self,
        code_hash: str,
        env_hash: str,
        dataset_hash: str,
        hyperparams: HyperparameterBundle,
        metrics: MetricBundle,
    ) -> str:
        # Use a separator that cannot appear in hex strings
        parts = [
            ("code_hash", code_hash),
            ("env_hash", env_hash),
            ("dataset_hash", dataset_hash),
            ("hyperparams", self._serializer.to_json(hyperparams)),
            ("metrics", self._serializer.to_json(metrics)),
        ]
        
        canonical = dict(parts)
        json_str = self._serializer.to_json(canonical)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def compute_version_hash(self, training_run_hash: str, parent_version_hash: str | None) -> str:
        data = {
            "training_run_hash": training_run_hash,
            "parent_version_hash": parent_version_hash,
        }
        json_str = self._serializer.to_json(data)
        return hashlib.sha256(json_str.encode()).hexdigest()