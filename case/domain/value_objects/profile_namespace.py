from dataclasses import dataclass, field
from typing import Any, Dict


class MissingProfileMetricError(KeyError):
    """Raised when a required namespace metric is unavailable."""


@dataclass
class ProfileNamespace:
    name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def require_metric(self, metric: str) -> Any:
        if metric not in self.metrics:
            raise MissingProfileMetricError(
                f"Required metric '{metric}' missing from namespace '{self.name}'"
            )
        return self.metrics[metric]
