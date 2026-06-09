from abc import ABC
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from core.domain.entities.finding import Finding
from core.domain.entities.health_score import HealthScore
from core.domain.entities.recommendation import Recommendation
from core.domain.enums.investigation_engine import InvestigationEngine
from core.domain.enums.severity import Severity

class Report(ABC, BaseModel)  : 
    """
    Abstract base class for all health reports produced by investigation engines.

    This provides a common interface for DataHealthReport, ModelHealthReport,
    and SystemHealthReport. Each subclass defines its own engine identifier
    and may add additional domain-specific fields.

    Attributes:
        id: Unique report identifier.
        engine: Which investigation engine produced this report
            ("data_investigation", "model_investigation", "system_investigation").
        findings: All findings discovered during the investigation.
        health_score: Computed health score based on finding severities.
        recommendations: Aggregated, deduplicated recommendations from all findings.
        generated_at: When the report was produced.
        dataset_id: The dataset this report covers, if applicable.
        model_id: The model this report covers, if applicable.
    """
        
    id : UUID = uuid4()
    engine : InvestigationEngine  = Field(default_factory=InvestigationEngine)
    findings : List[Finding] = Field(default_factory=list)
    health_score : HealthScore = Field(default_factory= HealthScore)
    recommendations : List[Recommendation] = Field (default_factory= list)
    generated_at : datetime = Field(default_factory=datetime.now)
    dataset_id : Optional[str] = None
    model_id : Optional[str] = None
    
    def top_findings(self, n: int = 5) -> List[Finding]:
        """
        Return the top N most severe findings.

        Findings are sorted by severity (CRITICAL first), then by
        confidence (highest first) within each severity level.

        Args:
            n: Maximum number of findings to return.

        Returns:
            Ordered list of up to n findings.
        """
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        sorted_findings = sorted(
            self.findings,
            key=lambda f: (
                severity_order.get(f.severity, 5),
                -f.confidence
            )
        )
        return sorted_findings[:n]
    
    def critical_findings(self) -> List[Finding]:
        """
        Return all findings with CRITICAL or HIGH severity.

        Returns:
            List of critical and high-severity findings.
        """
        return [
            f for f in self.findings
            if f.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

    



