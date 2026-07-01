from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ProfileNamespace:
    name : str
    metrics : Dict[str, Any] = field(default_factory=dict)
    artifacts :Dict[str, Any] = field(default_factory=dict)
    metadata : Dict[str, Any] = field(default_factory=dict)