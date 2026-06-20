from __future__ import annotations
from dataclasses import dataclass , field
from datetime import datetime
from typing import  Dict, Optional
from ulid import ulid

from pydantic import BaseModel
from core.evidence.evidence import Evidence
from core.observation.observation import Observation
from reasoning_graph.graph.nodes.node_status import NodeStatus
from reasoning_graph.graph.nodes.node_type import NodeType
from core.evidence.evidence_type import EvidenceType

class BaseNode(BaseModel):
    """
    Abstract base for every node in the reasoning graph.
    All nodes share identity, timing, and status tracking.
    """
    id:           str               
    
    node_type:    NodeType       
    status:       NodeStatus     = NodeStatus.INACTIVE

    label:        str                      
    description:  str            
    
    created_at:   datetime       = field(default_factory=datetime.now)
    updated_at:   datetime       = field(default_factory=datetime.now)
    
    metadata:     Dict           = field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if self.id is None:
            self.id = str(ulid())
        self.touch()
    
    def touch(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = datetime.now()



class ObservationNode(BaseNode):
    """
    Represents a single piece of observation collected from
    any subsystem (profiling, leakage, label quality, etc.).
    """
    node_type:          NodeType  = NodeType.OBSERVATION
    observation : Observation


class PatternNode(BaseNode):
    """
    A recognized semantic structure sitting between evidences and
    hypotheses. Carries graded confidence, computed by the same
    belief-propagation machinery as a HypothesisNode — there is no
    boolean present/absent state.

    level distinguishes primitive (L1) / structural (L2) / behavioral
    (L3) patterns, purely for organization and UI grouping — it does
    NOT change how propagation works, since edges already encode the
    actual dependency structure.
    """
    node_type:       NodeType  = NodeType.PATTERN   # new enum member, see below
    pattern_type:    str       = ""    # e.g. "LocalizedAnomaly", "PredictiveMissingness"
    level:           int       = 1     # 1 = primitive, 2 = structural, 3 = behavioral
    claim:           str       = ""    # "income missingness is concentrated in new accounts"
    subject:         "Subject" = None  # what the pattern is about

    confidence:          float = 0.0   # posterior, same semantics as HypothesisNode.confidence
    confidence_history:  list  = field(default_factory=list)

    # Provenance — what fed this pattern (observations and/or lower patterns)
    composed_from:       list[str] = field(default_factory=list)  # node IDs
    recognizer_id:        str       = ""

    status:    NodeStatus = NodeStatus.ACTIVE  # patterns skip DORMANT; they
                                                # exist the moment a recognizer fires


class HypothesisNode(BaseNode):
    """
    The central reasoning unit. A candidate explanation for
    one or more observations, with a probabilistic confidence
    score updated via belief propagation.
    """
    node_type:          NodeType  = NodeType.HYPOTHESIS
    hypothesis_type:    str       = ""   # e.g. "TEMPORAL_LEAKAGE"
    claim:              str       = ""   # "Feature X leaks future info because..."
    mechanism:          str       = ""   # HOW the cause produces the effect
    subject:            str       = ""   # what is being explained

    # Probabilistic state
    prior:              float     = 0.5  # initial probability before evidence
    confidence:         float     = 0.5  # current posterior P(H | all evidence)
    confidence_history: list      = field(default_factory=list)  # [(timestamp, value)]

    # Evidence accounting
    supporting_evidence:    list[str] = field(default_factory=list)  # ObservationNode IDs
    contradicting_evidence: list[str] = field(default_factory=list)  # ObservationNode IDs
    evidence_requests:      list[str] = field(default_factory=list)  # EvidenceRequest IDs

    # Lifecycle
    born_via:    str       = ""    # TEMPLATE | ABDUCTIVE | CASCADE | USER
    template_id: str       = ""    # source template if born via template
    parent_id:   Optional[str] = None  # parent hypothesis if born via cascade
    resolved_at: Optional[datetime] = None

    # Impact
    estimated_performance_impact: Optional[float] = None
    business_impact_score:        Optional[float] = None


@dataclass
class ContextNode(BaseNode):
    """
    Encodes background knowledge that modulates hypothesis priors.
    Examples: dataset type, ML task, domain, time period, 
    known business rules.

    Context nodes shift what is plausible before any evidence
    arrives — a financial time-series dataset makes temporal
    leakage far more likely a priori than a cross-sectional 
    image dataset.
    """
    node_type:    NodeType = NodeType.CONTEXT
    context_type: str      = ""   # "DATASET_TYPE", "DOMAIN", "TASK_TYPE"
    value:        str      = ""   # "time_series", "finance", "classification"
    prior_modifiers: dict  = field(default_factory=dict)
    # e.g. {"TEMPORAL_LEAKAGE": +0.25, "ANNOTATOR_DISAGREEMENT": -0.10}



class InterventionNode(BaseNode):
    """
    Represents a counterfactual action — a what-if scenario
    that can be simulated to validate or reject a hypothesis.
    
    Example: "Remove feature X and re-evaluate model" is an
    intervention that, if simulated, provides evidence about
    whether X is a shortcut feature.
    """
    node_type:          NodeType  = NodeType.INTERVENTION
    intervention_type:  str       = ""   # "FEATURE_REMOVAL", "RETRAIN", "RECALIBRATE"
    target:             str       = ""   # what the intervention acts on
    simulated:          bool      = False
    simulation_result:  Optional[dict] = None
    estimated_impact:   Optional[float] = None
    triggered_by:       str       = ""   # HypothesisNode ID



class FindingNode(BaseNode):
    """
    A confirmed hypothesis that has been promoted to a Finding.
    Carries the full reasoning trace — the evidence chain and
    graph path that led to this conclusion.
    """
    node_type:        NodeType   = NodeType.FINDING
    finding_type:     str        = ""
    severity:         str        = ""   # CRITICAL | HIGH | MEDIUM | LOW
    confidence:       float      = 0.0
    source_hypothesis: str       = ""   # HypothesisNode ID it graduated from
    reasoning_trace:  list[dict] = field(default_factory=list)
    # [{"node_id": ..., "node_type": ..., "contribution": float}]
    recommendations:  list[str]  = field(default_factory=list)
    estimated_fix_impact: Optional[float] = None