from pydantic import BaseModel, Field
from typing import Callable,Set
from core.shared.domain.enums.rule_severity import RuleSeverity

class ValidationRule(BaseModel):
    """Unified rule definition"""
    name: str
    func: Callable
    dependencies: Set[str] = Field(default_factory=set)
    severity: RuleSeverity = RuleSeverity.BLOCKING 
    description: str = ""