
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class InvestigationContext:
    """
    Immutable context passed into every subsystem during an investigation run.

    This object carries everything a subsystem needs to know about the
    investigation: what's being analyzed, under what configuration, at
    what capability tier, and who or what triggered the run.

    It is immutable — use with_config() to create a modified copy.

    Attributes:
        run_id: Unique identifier for this investigation run.
        dataset_id: The dataset being investigated, if any.
        model_id: The model being investigated, if any.
        config: Configuration dict controlling analysis behavior
            (thresholds, sampling rates, feature lists).
        triggered_by: What triggered this run
            (e.g., "user", "DatasetRegisteredEvent", "scheduler").
    """
    run_id: UUID = field(default_factory=uuid4)
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    triggered_by: str = "user"

    def with_config(self, key: str, value: Any) -> InvestigationContext:
        """
        Return a new InvestigationContext with one config value updated.

        The original context is not modified. This creates a shallow copy
        with the specified key set to the new value.

        Args:
            key: Configuration key to update.
            value: New value for the key.

        Returns:
            A new InvestigationContext with the updated config.
        """
        new_config = {**self.config, key: value}
        return replace(self, config=new_config)