from pydantic import BaseModel, Field , ConfigDict
from typing import Any , Optional, Dict
from core.shared.domain.enums.warning_severity import WarningSeverity

class ValidationWarning(BaseModel):
    """
    A non-blocking validation warning for a CodeSnapshot field.
    Includes severity to guide deployment decisions.
    """
    code : Any
    detail : str
    severity : WarningSeverity
    context : Optional[Dict[str, Any]] = Field(None)
    model_config= ConfigDict(frozen=True)