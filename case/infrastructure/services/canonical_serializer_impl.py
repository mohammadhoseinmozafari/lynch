
import json
from typing import Any
from case.domain.services.canonical_serializer import CanonicalSerializer

class CanonicalSerializerImpl(CanonicalSerializer):
    def to_json(self, obj: Any) -> str:
        # Use dataclasses / Pydantic models: convert to dict first if needed
        if hasattr(obj, "model_dump"):
            d = obj.model_dump(mode="json", exclude_none=True)
        elif hasattr(obj, "__dict__"):
            d = obj.__dict__
        else:
            d = obj
        return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)