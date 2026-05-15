from typing import (
    Any,
    Optional,
    Dict
)
from pydantic import BaseModel, Field, ConfigDict
class ValidationError(BaseModel):
    """
    A blocking validation error.
    Frozen (immutable) so errors can be safely collected and passed around.
    """
    code: Any
    detail: str
    context : Optional[Dict[str, Any]] = Field(None)
    model_config= ConfigDict(frozen=True)
