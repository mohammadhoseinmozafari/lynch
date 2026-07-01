from abc import ABC, abstractmethod
from typing import Any

class CanonicalSerializer(ABC):
    @abstractmethod
    def to_json(self, obj: Any) -> str:
        """Serialize any value object / DTO to a canonical JSON string (sorted keys, no extra spaces)."""
        pass