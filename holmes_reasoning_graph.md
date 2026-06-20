# Holmes — Reasoning Graph: Full Module & Class Design

---

## Module Overview

The Reasoning Graph is decomposed into 7 modules, each with a single responsibility:

```
holmes/
└── reasoning_graph/
    ├── graph/
    │   ├── nodes.py          # All node type definitions
    │   ├── edges.py          # All edge type definitions
    │   └── graph.py          # Core graph structure & operations
    ├── inference/
    │   ├── propagator.py     # Belief propagation engine
    │   ├── conflict.py       # Conflict detection & resolution
    │   └── decay.py          # Temporal confidence decay
    ├── construction/
    │   ├── builder.py        # Graph construction orchestrator
    │   ├── template_loader.py# Hypothesis template library loader
    │   └── abductive.py      # LLM-based hypothesis generation
    ├── hypothesis/
    │   ├── hypothesis.py     # Hypothesis lifecycle management
    │   ├── lifecycle.py      # State machine for hypothesis states
    │   └── spawner.py        # Cascade hypothesis spawning
    ├── evidence/
    │   ├── evidence.py       # Evidence node management
    │   ├── matcher.py        # Evidence → template matching
    │   └── requester.py      # Active evidence request emission
    ├── query/
    │   ├── tracer.py         # Causal path tracing
    │   ├── ranker.py         # Hypothesis ranking & scoring
    │   └── exporter.py       # Graph → Finding export
    └── store/
        ├── graph_store.py    # Graph persistence (Neo4j adapter)
        └── event_log.py      # Event-sourced change log
```

---

## Module 1: `graph/` — Core Graph Structure

### `nodes.py`

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


class NodeType(Enum):
    OBSERVATION  = "observation"   # raw evidence / measurement
    HYPOTHESIS   = "hypothesis"    # candidate explanation
    CONTEXT      = "context"       # background knowledge / metadata
    INTERVENTION = "intervention"  # counterfactual / what-if action
    FINDING      = "finding"       # graduated, confirmed conclusion


class NodeStatus(Enum):
    DORMANT      = "dormant"       # exists but not yet activated
    ACTIVE       = "active"        # currently being evaluated
    CONFIRMED    = "confirmed"     # confidence above threshold
    REJECTED     = "rejected"      # confidence below threshold
    INCONCLUSIVE = "inconclusive"  # evidence conflict, needs more info
    MERGED       = "merged"        # unified with another node
    GRADUATED    = "graduated"     # promoted to a Finding


@dataclass
class BaseNode:
    """
    Abstract base for every node in the reasoning graph.
    All nodes share identity, timing, and status tracking.
    """
    id:           str            = field(default_factory=lambda: str(uuid.uuid4()))
    node_type:    NodeType       = None
    label:        str            = ""          # human-readable name
    description:  str            = ""
    status:       NodeStatus     = NodeStatus.DORMANT
    created_at:   datetime       = field(default_factory=datetime.utcnow)
    updated_at:   datetime       = field(default_factory=datetime.utcnow)
    metadata:     dict           = field(default_factory=dict)

    def touch(self):
        """Update the modified timestamp."""
        self.updated_at = datetime.utcnow()


@dataclass
class ObservationNode(BaseNode):
    """
    Represents a single piece of evidence collected from
    any subsystem (profiling, leakage, label quality, etc.).

    Reliability score models how trustworthy the measurement
    method is — a statistical test on 100k rows scores higher
    than a heuristic pattern match on 50 rows.
    """
    node_type:          NodeType  = NodeType.OBSERVATION
    evidence_type:      str       = ""    # e.g. "MISSING_RATE", "KS_STATISTIC"
    value:              float     = None  # numeric result of the measurement
    value_raw:          dict      = field(default_factory=dict)  # full payload
    reliability:        float     = 1.0  # [0,1] — how trustworthy is this source
    collected_at:       datetime  = field(default_factory=datetime.utcnow)
    collector_id:       str       = ""   # which subsystem produced it
    subject:            str       = ""   # feature name / segment / model layer
    supports_types:     list[str] = field(default_factory=list)  # hypothesis types it can activate


@dataclass
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


@dataclass
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


@dataclass
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
```

---

### `edges.py`

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class EdgeType(Enum):
    EVIDENTIAL    = "evidential"    # ObservationNode → HypothesisNode
    CAUSAL        = "causal"        # HypothesisNode → HypothesisNode (cause → effect)
    COMPETITIVE   = "competitive"   # HypothesisNode ↔ HypothesisNode (mutual exclusion)
    CORROBORATING = "corroborating" # HypothesisNode → HypothesisNode (non-causal support)
    CONTEXTUAL    = "contextual"    # ContextNode → HypothesisNode (prior shift)
    PREREQUISITE  = "prerequisite"  # HypothesisNode → HypothesisNode (must resolve first)
    SPAWNED       = "spawned"       # HypothesisNode → HypothesisNode (parent → child cascade)


@dataclass
class BaseEdge:
    """
    Every edge in the reasoning graph.
    The weight encodes the strength of the relationship —
    its exact meaning depends on EdgeType (see subclasses).
    """
    id:          str      = field(default_factory=lambda: str(uuid.uuid4()))
    edge_type:   EdgeType = None
    source_id:   str      = ""   # source node ID
    target_id:   str      = ""   # target node ID
    weight:      float    = 1.0  # relationship strength [0,1]
    created_at:  datetime = field(default_factory=datetime.utcnow)
    metadata:    dict     = field(default_factory=dict)


@dataclass
class EvidentialEdge(BaseEdge):
    """
    Connects an ObservationNode to a HypothesisNode.

    p_e_given_h:     P(E | H)  — how likely is this evidence if H is true?
    p_e_given_not_h: P(E | ¬H) — how likely is this evidence if H is false?

    The likelihood ratio p_e_given_h / p_e_given_not_h is the
    core update multiplier in Bayes' rule. A high ratio means
    this evidence is highly diagnostic for this hypothesis.
    """
    edge_type:        EdgeType = EdgeType.EVIDENTIAL
    p_e_given_h:      float    = 0.5   # P(E | H)
    p_e_given_not_h:  float    = 0.5   # P(E | ¬H)
    polarity:         str      = "supports"  # "supports" | "contradicts"

    @property
    def likelihood_ratio(self) -> float:
        if self.p_e_given_not_h == 0:
            return float('inf')
        return self.p_e_given_h / self.p_e_given_not_h


@dataclass
class CausalEdge(BaseEdge):
    """
    Connects two HypothesisNodes where source causes target.
    
    p_target_given_source:     P(B | A=true)
    p_target_given_not_source: P(B | A=false)

    Example: DATA_PIPELINE_FAILURE → FEATURE_DRIFT
    If the pipeline fails (A=true), feature drift is very likely.
    If the pipeline is healthy (A=false), drift is rare.
    """
    edge_type:                  EdgeType = EdgeType.CAUSAL
    p_target_given_source:      float    = 0.8  # P(effect | cause is true)
    p_target_given_not_source:  float    = 0.1  # P(effect | cause is false)
    causal_lag_seconds:         Optional[int] = None  # time delay between cause and effect


@dataclass
class CompetitiveEdge(BaseEdge):
    """
    Connects two HypothesisNodes that are mutually exclusive
    explanations of the same observation (explaining away).

    suppression_strength: how strongly confirming one
    suppresses the other. 1.0 = fully mutually exclusive,
    0.0 = no suppression (they can coexist).

    Example: LOW_MODEL_CAPACITY ↔ INSUFFICIENT_TRAINING_DATA
    — both explain poor performance, but remedies differ.
    """
    edge_type:            EdgeType = EdgeType.COMPETITIVE
    suppression_strength: float    = 0.8   # [0,1]
    shared_evidence_ids:  list[str] = field(default_factory=list)


@dataclass
class CorroboratingEdge(BaseEdge):
    """
    Non-causal: source being true makes target more plausible.
    Weaker than causal — no mechanism, just co-occurrence pattern
    from prior knowledge.
    """
    edge_type:          EdgeType = EdgeType.CORROBORATING
    corroboration_strength: float = 0.3  # how much confidence boost target gets


@dataclass
class ContextualEdge(BaseEdge):
    """
    From ContextNode to HypothesisNode.
    Shifts the prior of the hypothesis based on background knowledge.

    prior_delta: +0.25 means "in this context, this hypothesis
    starts 25 percentage points more likely than the default prior."
    """
    edge_type:    EdgeType = EdgeType.CONTEXTUAL
    prior_delta:  float    = 0.0   # additive shift to hypothesis prior


@dataclass
class PrerequisiteEdge(BaseEdge):
    """
    Source must reach a minimum confidence before target activates.
    Models logical dependencies between hypotheses.

    Example: LEAKAGE_SUSPECTED must be ACTIVE before
    TEMPORAL_LEAKAGE child hypotheses are worth evaluating.
    """
    edge_type:           EdgeType = EdgeType.PREREQUISITE
    activation_threshold: float   = 0.5  # min confidence of source to unlock target


@dataclass
class SpawnedEdge(BaseEdge):
    """
    Created when a confirmed hypothesis spawns child hypotheses
    via cascade. Tracks the parent-child lineage.
    """
    edge_type:    EdgeType = EdgeType.SPAWNED
    spawn_reason: str      = ""   # why this child was spawned
```

---

### `graph.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

from .nodes import BaseNode, HypothesisNode, ObservationNode, NodeStatus
from .edges import BaseEdge, EdgeType


@dataclass
class ReasoningGraph:
    """
    The core graph structure for a single investigation.
    Holds all nodes and edges, exposes structural queries,
    and coordinates with the inference engine.

    One ReasoningGraph is created per investigation session
    (e.g., per dataset registration, per production drift event).
    Multiple graphs are connected via the MetaGraph.
    """
    id:             str       = field(default_factory=lambda: str(uuid.uuid4()))
    investigation_id: str     = ""
    created_at:     datetime  = field(default_factory=datetime.utcnow)
    updated_at:     datetime  = field(default_factory=datetime.utcnow)

    # Core storage
    nodes:  dict[str, BaseNode] = field(default_factory=dict)  # id → node
    edges:  dict[str, BaseEdge] = field(default_factory=dict)  # id → edge

    # Adjacency for fast traversal
    # outgoing[node_id] = list of edge IDs leaving that node
    # incoming[node_id] = list of edge IDs entering that node
    outgoing: dict[str, list[str]] = field(default_factory=dict)
    incoming: dict[str, list[str]] = field(default_factory=dict)

    # Indexes for fast lookup
    nodes_by_type:   dict[str, list[str]] = field(default_factory=dict)
    nodes_by_status: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, node: BaseNode) -> str:
        """
        Add a node to the graph.
        Initializes adjacency entries and updates indexes.
        Returns the node ID.
        """

    def add_edge(self, edge: BaseEdge) -> str:
        """
        Add a directed edge between two existing nodes.
        Validates that both endpoint nodes exist.
        Updates outgoing/incoming adjacency maps.
        Returns the edge ID.
        """

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all edges connected to it.
        Used for pruning rejected or merged nodes.
        """

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Return a node by ID, or None if not found."""

    def get_edge(self, edge_id: str) -> Optional[BaseEdge]:
        """Return an edge by ID, or None if not found."""

    def neighbors(self, node_id: str,
                  edge_type: Optional[EdgeType] = None,
                  direction: str = "outgoing") -> list[BaseNode]:
        """
        Return neighboring nodes, optionally filtered by edge type.
        direction: "outgoing" | "incoming" | "both"
        """

    def get_edges_between(self, source_id: str,
                           target_id: str) -> list[BaseEdge]:
        """Return all edges connecting two specific nodes."""

    def subgraph(self, node_ids: list[str]) -> "ReasoningGraph":
        """
        Return a new ReasoningGraph containing only the
        specified nodes and the edges between them.
        Used for focused sub-investigations.
        """

    def active_hypotheses(self) -> list[HypothesisNode]:
        """Return all hypothesis nodes currently in ACTIVE status."""

    def confirmed_hypotheses(self) -> list[HypothesisNode]:
        """Return all CONFIRMED hypothesis nodes."""

    def inconclusive_hypotheses(self) -> list[HypothesisNode]:
        """Return all INCONCLUSIVE hypothesis nodes (need more evidence)."""

    def unattached_observations(self) -> list[ObservationNode]:
        """
        Return observation nodes with no active hypothesis connections.
        These are candidates for the abductive engine.
        """

    def merge_nodes(self, node_id_a: str, node_id_b: str,
                    merged_label: str) -> str:
        """
        Merge two hypothesis nodes that converge on the same
        explanation. Redirects all edges to a new merged node.
        Marks originals as MERGED.
        Returns the new merged node ID.
        """

    def snapshot(self) -> dict:
        """
        Serialize the entire graph state to a dictionary.
        Used by the event log and graph store for persistence.
        """

    def diff(self, other: "ReasoningGraph") -> dict:
        """
        Compute the structural and belief-state diff between
        this graph and another version. Used for change tracking.
        """
```

---

## Module 2: `inference/` — Belief Propagation

### `propagator.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..graph.graph import ReasoningGraph
from ..graph.nodes import HypothesisNode, ObservationNode
from ..graph.edges import (EvidentialEdge, CausalEdge,
                           CompetitiveEdge, CorroboratingEdge)


@dataclass
class PropagationResult:
    """
    The outcome of a single belief propagation run.
    Records which nodes changed and by how much.
    """
    run_id:          str       = ""
    triggered_by:    str       = ""   # evidence ID or "manual"
    started_at:      datetime  = field(default_factory=datetime.utcnow)
    completed_at:    Optional[datetime] = None
    iterations:      int       = 0
    converged:       bool      = False
    updated_nodes:   list[str] = field(default_factory=list)  # node IDs that changed
    delta_map:       dict[str, float] = field(default_factory=dict)
    # node_id → confidence change (e.g. "h_123": +0.18)
    explaining_away_events: list[dict] = field(default_factory=list)
    # [{"suppressed": node_id, "by": node_id, "delta": float}]


class BeliefPropagator:
    """
    Implements iterative loopy belief propagation on the
    reasoning graph. When new evidence arrives, beliefs
    flow through the graph until convergence.

    Uses message-passing: each node sends a 'message'
    encoding its current belief to each neighbor. Messages
    are iteratively updated until delta < convergence_threshold.

    For competitive edges, the explaining-away mechanism
    is applied: when one hypothesis gains confidence, its
    competitors lose confidence proportionally.
    """

    def __init__(self,
                 graph: ReasoningGraph,
                 convergence_threshold: float = 0.001,
                 max_iterations: int = 100,
                 damping_factor: float = 0.5):
        self.graph = graph
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.damping_factor = damping_factor
        # message_buffer[node_id][neighbor_id] = float
        self.message_buffer: dict[str, dict[str, float]] = {}

    def propagate(self, triggered_by: str) -> PropagationResult:
        """
        Main entry point. Run full belief propagation across
        the graph until convergence or max_iterations.
        Returns a PropagationResult with the full change record.
        """

    def _initialize_messages(self) -> None:
        """
        Initialize all messages to 1.0 (neutral).
        Called at the start of each propagation run.
        """

    def _compute_evidential_message(self,
                                     obs: ObservationNode,
                                     hyp: HypothesisNode,
                                     edge: EvidentialEdge) -> float:
        """
        Compute the likelihood ratio message from an observation
        node to a hypothesis node.

        message = P(E | H) / P(E | ¬H) × reliability_weight

        reliability_weight = obs.reliability (discounts noisy sources)
        """

    def _compute_causal_message(self,
                                 source_hyp: HypothesisNode,
                                 target_hyp: HypothesisNode,
                                 edge: CausalEdge) -> float:
        """
        Propagate confidence through a causal edge.

        new_prior(target) = P(target | source=true) × conf(source)
                          + P(target | source=false) × (1 - conf(source))

        This is the standard total probability law update.
        """

    def _apply_explaining_away(self,
                                updated_hyp: HypothesisNode) -> list[dict]:
        """
        After updating a hypothesis, suppress its competitors
        via competitive edges.

        suppressed_conf = competitor.confidence
                        × (1 - edge.suppression_strength
                           × updated_hyp.confidence)

        Returns list of explaining-away events for the result log.
        """

    def _apply_corroboration(self,
                              source_hyp: HypothesisNode) -> None:
        """
        After a hypothesis updates, boost corroborated neighbors
        by a small damped factor.

        boost = edge.corroboration_strength × source.confidence × damping_factor
        """

    def _update_node_confidence(self,
                                 hyp: HypothesisNode,
                                 new_confidence: float) -> float:
        """
        Apply damped update to avoid oscillation:
        
        conf_new = (1 - damping_factor) × conf_old
                 + damping_factor × new_confidence

        Records the update in confidence_history.
        Returns the absolute delta.
        """

    def _has_converged(self, deltas: list[float]) -> bool:
        """
        Return True when all confidence deltas in this
        iteration are below convergence_threshold.
        """

    def partial_propagate(self, affected_node_ids: list[str],
                           triggered_by: str) -> PropagationResult:
        """
        Optimized propagation that only updates the subgraph
        reachable from the affected nodes. Used when evidence
        targets a specific feature or segment, avoiding a full
        graph re-computation.
        """
```

---

### `conflict.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..graph.graph import ReasoningGraph
from ..graph.nodes import HypothesisNode, ObservationNode


class ConflictResolutionStrategy(Enum):
    RELIABILITY_WINS  = "reliability_wins"   # higher-reliability evidence wins
    RECENCY_WINS      = "recency_wins"       # more recent evidence wins
    ACTIVE_INQUIRY    = "active_inquiry"     # request discriminating evidence
    SPLIT_HYPOTHESIS  = "split_hypothesis"   # create two competing child hypotheses


@dataclass
class EvidenceConflict:
    """Represents a detected conflict between two pieces of evidence."""
    hypothesis_id:       str
    evidence_a_id:       str   # supports hypothesis
    evidence_b_id:       str   # contradicts hypothesis
    evidence_a_weight:   float
    evidence_b_weight:   float
    magnitude:           float  # how strong the conflict is [0,1]
    detected_at:         str
    resolution:          Optional[ConflictResolutionStrategy] = None
    resolution_result:   Optional[dict] = None


class ConflictDetector:
    """
    Scans the reasoning graph for evidence conflicts —
    cases where two observations push a hypothesis's
    confidence in strongly opposing directions.

    A conflict is flagged when:
      abs(P(H|E_a) - P(H|E_b)) > conflict_threshold
    """

    def __init__(self, graph: ReasoningGraph, conflict_threshold: float = 0.4):
        self.graph = graph
        self.conflict_threshold = conflict_threshold

    def scan(self) -> list[EvidenceConflict]:
        """
        Scan all active hypotheses for evidence conflicts.
        Returns a list of detected conflicts.
        """

    def detect_for_hypothesis(self,
                               hyp: HypothesisNode) -> list[EvidenceConflict]:
        """
        Check a single hypothesis for conflicts among its
        supporting and contradicting evidence nodes.
        """

    def _compute_conflict_magnitude(self,
                                     evidence_a: ObservationNode,
                                     evidence_b: ObservationNode,
                                     hyp: HypothesisNode) -> float:
        """
        Compute how strongly E_a and E_b conflict for hypothesis H.
        Uses the absolute difference in posterior updates each
        evidence would produce independently.
        """


class ConflictResolver:
    """
    Resolves detected conflicts using the most appropriate
    strategy, selected based on evidence properties and
    hypothesis state.
    """

    def __init__(self, graph: ReasoningGraph):
        self.graph = graph

    def resolve(self, conflict: EvidenceConflict) -> ConflictResolutionStrategy:
        """
        Select and apply the best resolution strategy.
        Selection logic:
          - Large reliability gap → RELIABILITY_WINS
          - Large recency gap (> 7 days) → RECENCY_WINS
          - Neither → ACTIVE_INQUIRY if discriminating evidence exists
          - Otherwise → SPLIT_HYPOTHESIS
        """

    def apply_reliability_wins(self, conflict: EvidenceConflict) -> None:
        """Discount the lower-reliability evidence's edge weight."""

    def apply_recency_wins(self, conflict: EvidenceConflict) -> None:
        """Apply temporal decay to the older evidence's contribution."""

    def apply_active_inquiry(self, conflict: EvidenceConflict) -> str:
        """
        Emit an EvidenceRequest for a discriminating third piece
        of evidence that can adjudicate between E_a and E_b.
        Returns the EvidenceRequest ID.
        Sets hypothesis status to INCONCLUSIVE.
        """

    def apply_split_hypothesis(self, conflict: EvidenceConflict) -> tuple[str, str]:
        """
        Create two competing child hypotheses — one supported
        by E_a, one by E_b. Wire them with a competitive edge.
        Returns the IDs of the two new HypothesisNodes.
        """
```

---

### `decay.py`

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from ..graph.nodes import ObservationNode
from ..graph.edges import EvidentialEdge


@dataclass
class DecayConfig:
    """
    Configuration for temporal evidence decay.
    Controls how quickly older evidence loses influence.

    half_life_days: after this many days, evidence weight
                    is halved. Shorter = faster decay.
    floor:          minimum weight — evidence never fully
                    disappears, it just becomes negligible.
    decay_function: "exponential" | "linear" | "step"
    """
    half_life_days: float = 30.0
    floor:          float = 0.05
    decay_function: str   = "exponential"


class TemporalDecayEngine:
    """
    Applies time-based decay to evidence node weights,
    reducing the influence of stale observations on
    hypothesis confidence.

    This is critical for production monitoring: evidence
    of "no drift 6 months ago" should not suppress a
    current drift hypothesis.
    """

    def __init__(self, config: DecayConfig):
        self.config = config

    def current_weight(self,
                        evidence: ObservationNode,
                        reference_time: Optional[datetime] = None) -> float:
        """
        Compute the current effective weight of an evidence node,
        accounting for how old it is.
        
        Exponential decay formula:
          w(t) = max(floor, reliability × e^(-λ × Δt))
          where λ = ln(2) / half_life_days
        """

    def apply_decay_to_edge(self,
                             edge: EvidentialEdge,
                             evidence: ObservationNode) -> None:
        """
        Update an evidential edge's effective weight in-place
        based on the age of the source observation node.
        """

    def scan_and_decay(self, graph) -> list[str]:
        """
        Scan all observation nodes in the graph and apply
        decay to their connected evidential edges.
        Returns list of node IDs whose weights changed significantly.
        Called periodically (e.g., every hour for live investigations).
        """

    def _decay_function(self, age_days: float) -> float:
        """
        Compute the decay multiplier for a given age.
        Dispatches to the configured decay function type.
        """
```

---

## Module 3: `construction/` — Graph Builder

### `builder.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..graph.graph import ReasoningGraph
from ..graph.nodes import ContextNode, HypothesisNode, NodeStatus
from ..graph.edges import ContextualEdge
from .template_loader import TemplateLoader, HypothesisTemplate
from .abductive import AbductiveEngine


@dataclass
class GraphBuildContext:
    """
    All inputs available at the start of an investigation.
    Feeds into context node creation and template activation.
    """
    investigation_id: str
    dataset_type:     str       = ""   # "time_series" | "tabular" | "image"
    ml_task:          str       = ""   # "classification" | "regression" | "ranking"
    domain:           str       = ""   # "finance" | "healthcare" | "retail"
    has_temporal_features: bool = False
    has_multiple_annotators: bool = False
    model_type:       str       = ""
    known_issues:     list[str] = field(default_factory=list)
    custom_priors:    dict      = field(default_factory=dict)
    # hypothesis_type → override prior


class GraphBuilder:
    """
    Orchestrates the initial construction of a ReasoningGraph
    for a new investigation. 

    Phases:
    1. Create context nodes from investigation context
    2. Instantiate dormant hypothesis nodes from templates
    3. Wire contextual edges (context → hypothesis prior shifts)
    4. Wire prerequisite edges from template definitions
    5. Wire competitive and causal edges from template definitions
    """

    def __init__(self,
                 template_loader: TemplateLoader,
                 abductive_engine: AbductiveEngine):
        self.template_loader = template_loader
        self.abductive_engine = abductive_engine

    def build(self, context: GraphBuildContext) -> ReasoningGraph:
        """
        Main entry point. Build and return a fully initialized
        ReasoningGraph for the given investigation context.
        """

    def _create_context_nodes(self,
                               graph: ReasoningGraph,
                               context: GraphBuildContext) -> list[str]:
        """
        Create ContextNodes for dataset type, ML task, domain,
        and any custom context flags. Return list of node IDs.
        """

    def _instantiate_templates(self,
                                graph: ReasoningGraph,
                                context: GraphBuildContext) -> list[str]:
        """
        Load all hypothesis templates relevant to this context.
        Instantiate each as a DORMANT HypothesisNode.
        Apply custom prior overrides from context.
        Return list of node IDs created.
        """

    def _wire_contextual_edges(self,
                                graph: ReasoningGraph,
                                context_node_ids: list[str],
                                hypothesis_node_ids: list[str]) -> None:
        """
        For each context node, apply its prior_modifiers to
        matching hypothesis nodes via ContextualEdges.
        """

    def _wire_structural_edges(self,
                                graph: ReasoningGraph,
                                templates: list[HypothesisTemplate]) -> None:
        """
        Wire causal, competitive, prerequisite, and corroborating
        edges between hypothesis nodes, as defined in templates.
        """

    def integrate_evidence(self,
                            graph: ReasoningGraph,
                            evidence_batch: list) -> list[str]:
        """
        Add a batch of new evidence nodes to the graph.
        For each evidence node:
          1. Run template matcher to find matching hypotheses
          2. Create EvidentialEdges with calibrated likelihood ratios
          3. Activate matching DORMANT hypothesis nodes
          4. Pass unmatched evidence to AbductiveEngine
        Returns IDs of newly activated hypothesis nodes.
        """
```

---

### `template_loader.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ActivationCondition:
    """
    A single condition that evidence must satisfy to
    activate a hypothesis template.
    All conditions in a template's list are AND-ed;
    separate templates can model OR logic.
    """
    evidence_type:  str            # e.g. "FEATURE_TARGET_CORRELATION"
    operator:       str            # "gt" | "lt" | "eq" | "contains" | "exists"
    value:          Optional[float | str | bool] = None
    subject_filter: Optional[str] = None  # e.g. "temporal_features_only"


@dataclass
class HypothesisTemplate:
    """
    The blueprint for a known ML failure mode.
    Instantiated into a HypothesisNode when activation
    conditions are met.

    This is your institutional knowledge made executable.
    Each template encodes:
    - How to recognize the failure (activation_conditions)
    - How likely it is before any evidence (base_prior, context_priors)
    - What evidence confirms or rules it out (evidence requirements)
    - How it relates to other hypotheses (structural relationships)
    """
    id:             str
    hypothesis_type: str
    label:          str
    description:    str
    mechanism:      str   # HOW the cause produces the effect

    # Activation
    activation_conditions: list[ActivationCondition]
    applicable_contexts:   list[str] = field(default_factory=list)
    # ["time_series", "classification"] — empty = all contexts

    # Priors
    base_prior:      float = 0.3   # default prior
    context_priors:  dict  = field(default_factory=dict)
    # {"time_series": 0.55, "image": 0.05}

    # Evidence calibration
    # Maps evidence_type → (P(E|H), P(E|¬H))
    likelihood_ratios: dict[str, tuple[float, float]] = field(default_factory=dict)

    confirmation_evidence_types: list[str] = field(default_factory=list)
    rejection_evidence_types:    list[str] = field(default_factory=list)

    # Structural relationships (referenced by hypothesis_type strings)
    causes:          list[str] = field(default_factory=list)
    caused_by:       list[str] = field(default_factory=list)
    competes_with:   list[str] = field(default_factory=list)
    corroborates:    list[str] = field(default_factory=list)
    prerequisite_of: list[str] = field(default_factory=list)
    spawns_on_confirm: list[str] = field(default_factory=list)

    # Claim template (filled in on instantiation)
    claim_template: str = ""
    # e.g. "Feature {subject} leaks future info via {mechanism}"

    # Finding graduation
    confirmation_threshold: float = 0.80
    rejection_threshold:    float = 0.15
    severity:               str   = "MEDIUM"


class TemplateLoader:
    """
    Loads, validates, and indexes the hypothesis template library.
    Templates can be loaded from YAML files, a database, or
    registered programmatically.

    The template library is the primary knowledge base of Holmes —
    every known ML failure mode lives here as a structured template.
    """

    def __init__(self, templates_path: str):
        self.templates_path = templates_path
        self._templates: dict[str, HypothesisTemplate] = {}

    def load_all(self) -> None:
        """Load all templates from the configured source."""

    def get(self, hypothesis_type: str) -> Optional[HypothesisTemplate]:
        """Retrieve a template by its hypothesis type string."""

    def get_for_context(self, context_flags: list[str]) -> list[HypothesisTemplate]:
        """
        Return all templates applicable to the given context flags.
        Empty applicable_contexts means universally applicable.
        """

    def register(self, template: HypothesisTemplate) -> None:
        """Programmatically register a new template at runtime."""

    def validate(self, template: HypothesisTemplate) -> list[str]:
        """
        Validate a template for internal consistency.
        Returns list of validation errors (empty = valid).
        Checks: referenced types exist, priors in [0,1],
        likelihood ratios valid, no circular prerequisites.
        """
```

---

### `abductive.py`

```python
from dataclasses import dataclass, field
from typing import Optional

from ..graph.nodes import ObservationNode, HypothesisNode
from ..graph.graph import ReasoningGraph


@dataclass
class AbductiveProposal:
    """
    A hypothesis proposed by the LLM-based abductive engine
    for an evidence cluster that matched no template.
    """
    claim:          str
    mechanism:      str
    hypothesis_type: str        # LLM-generated type label
    confidence:     float       # LLM's estimated plausibility [0,1]
    supporting_evidence_ids: list[str] = field(default_factory=list)
    proposed_edges: list[dict]  = field(default_factory=list)
    # [{"type": "COMPETITIVE", "target_type": "PROXY_VARIABLE"}]
    confirmation_evidence_needed: list[str] = field(default_factory=list)
    rejection_evidence_would_be:  list[str] = field(default_factory=list)
    llm_reasoning: str = ""     # raw explanation from LLM for auditability


class AbductiveEngine:
    """
    Generates novel hypothesis proposals for evidence clusters
    that don't match any template in the library.

    Uses an LLM to perform abductive reasoning:
    "Given these observations, what is the most likely explanation?"

    Proposals are treated with lower initial trust than
    template-derived hypotheses (prior deflated by novelty_discount).
    The LLM proposes; the graph and propagator decide.
    """

    def __init__(self,
                 llm_client,
                 novelty_discount: float = 0.2,
                 max_proposals_per_cluster: int = 3):
        self.llm_client = llm_client
        self.novelty_discount = novelty_discount
        self.max_proposals_per_cluster = max_proposals_per_cluster

    def propose(self,
                unmatched_evidence: list[ObservationNode],
                graph: ReasoningGraph,
                investigation_context: dict) -> list[AbductiveProposal]:
        """
        Main entry point. Cluster unmatched evidence, build
        prompts, call LLM, parse and validate proposals.
        Returns a list of AbductiveProposals.
        """

    def _cluster_evidence(self,
                           evidence: list[ObservationNode]) -> list[list[ObservationNode]]:
        """
        Group unmatched evidence nodes by similarity
        (subject, evidence_type, temporal proximity).
        Each cluster is reasoned about independently.
        """

    def _build_prompt(self,
                       cluster: list[ObservationNode],
                       graph: ReasoningGraph,
                       context: dict) -> str:
        """
        Build the abductive reasoning prompt for the LLM.
        Includes: investigation context, evidence summaries,
        existing confirmed hypotheses, and structured output spec.
        """

    def _parse_response(self, llm_response: str,
                         cluster: list[ObservationNode]) -> list[AbductiveProposal]:
        """
        Parse the LLM's JSON response into AbductiveProposal objects.
        Validates structure and clips confidence scores.
        """

    def _apply_novelty_discount(self,
                                 proposal: AbductiveProposal) -> AbductiveProposal:
        """
        Reduce the confidence of novel proposals by novelty_discount.
        Novel hypotheses start with less trust than template-derived ones.
        """

    def integrate_proposals(self,
                             proposals: list[AbductiveProposal],
                             graph: ReasoningGraph) -> list[str]:
        """
        Instantiate accepted proposals as ACTIVE HypothesisNodes
        in the graph, wire their edges, and return their IDs.
        """
```

---

## Module 4: `hypothesis/` — Lifecycle Management

### `lifecycle.py`

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..graph.nodes import HypothesisNode, NodeStatus


class LifecycleTransition(Enum):
    ACTIVATE    = "activate"     # DORMANT → ACTIVE
    CONFIRM     = "confirm"      # ACTIVE → CONFIRMED
    REJECT      = "reject"       # ACTIVE → REJECTED
    STALL       = "stall"        # ACTIVE → INCONCLUSIVE
    RESUME      = "resume"       # INCONCLUSIVE → ACTIVE
    MERGE       = "merge"        # ACTIVE → MERGED
    GRADUATE    = "graduate"     # CONFIRMED → GRADUATED (→ Finding)


@dataclass
class TransitionEvent:
    """Records every state change in a hypothesis's life."""
    hypothesis_id:  str
    transition:     LifecycleTransition
    from_status:    NodeStatus
    to_status:      NodeStatus
    reason:         str
    triggered_by:   str   # evidence ID, propagation run ID, or "user"
    occurred_at:    datetime = None


class HypothesisLifecycleManager:
    """
    Enforces the state machine for hypothesis nodes.
    Every status change goes through this class — no direct
    status mutations are allowed on hypothesis nodes.

    Valid transitions:
    DORMANT → ACTIVE (activation condition met)
    ACTIVE → CONFIRMED (confidence > template.confirmation_threshold)
    ACTIVE → REJECTED (confidence < template.rejection_threshold)
    ACTIVE → INCONCLUSIVE (evidence conflict detected)
    INCONCLUSIVE → ACTIVE (conflict resolved or new evidence)
    ACTIVE | CONFIRMED → MERGED (converges with another hypothesis)
    CONFIRMED → GRADUATED (promoted to Finding)
    """

    def __init__(self, graph, event_log):
        self.graph = graph
        self.event_log = event_log

    def transition(self,
                   hypothesis: HypothesisNode,
                   transition: LifecycleTransition,
                   reason: str,
                   triggered_by: str) -> TransitionEvent:
        """
        Attempt a state transition. Validates it is legal.
        Applies the transition if valid.
        Records the event in the event log.
        Returns the TransitionEvent.
        """

    def evaluate_all(self) -> list[TransitionEvent]:
        """
        Scan all ACTIVE hypotheses and apply any warranted
        automatic transitions based on current confidence levels.
        Called after each belief propagation run.
        """

    def _should_confirm(self, hyp: HypothesisNode) -> bool:
        """confidence > template.confirmation_threshold"""

    def _should_reject(self, hyp: HypothesisNode) -> bool:
        """confidence < template.rejection_threshold"""

    def _is_valid_transition(self,
                              from_status: NodeStatus,
                              to_status: NodeStatus) -> bool:
        """Check transition is in the allowed set."""
```

---

### `spawner.py`

```python
from dataclasses import dataclass, field
from typing import Optional

from ..graph.nodes import HypothesisNode
from ..graph.graph import ReasoningGraph
from .template_loader import TemplateLoader


@dataclass
class SpawnPlan:
    """Describes what child hypotheses a confirmed parent will spawn."""
    parent_id:    str
    child_types:  list[str]
    spawn_reason: str
    context_overrides: dict = field(default_factory=dict)
    # e.g. {"subject": "feature_group_A"} to specialize child


class HypothesisSpawner:
    """
    When a hypothesis reaches CONFIRMED status, it may spawn
    more specific child hypotheses for deeper investigation.

    This models the expert investigator's workflow:
    "We've confirmed there's leakage — now which type exactly?"

    Children are created as ACTIVE (not DORMANT), because the
    parent's confirmation is strong evidence for their relevance.
    Children inherit the parent's subject and context, refined
    by the spawn configuration in the parent's template.
    """

    def __init__(self,
                 graph: ReasoningGraph,
                 template_loader: TemplateLoader):
        self.graph = graph
        self.template_loader = template_loader

    def compute_spawn_plan(self,
                            confirmed_hyp: HypothesisNode) -> Optional[SpawnPlan]:
        """
        Look up the confirmed hypothesis's template.
        If spawns_on_confirm is non-empty, build a SpawnPlan.
        Returns None if no children should be spawned.
        """

    def execute_spawn(self, plan: SpawnPlan) -> list[str]:
        """
        Instantiate child HypothesisNodes from the spawn plan.
        Wire SpawnedEdges from parent to each child.
        Set children to ACTIVE status with elevated priors
        (parent confirmation boosts children's starting confidence).
        Return list of child node IDs.
        """

    def _compute_child_prior(self,
                              parent: HypothesisNode,
                              child_type: str) -> float:
        """
        Child's starting confidence = base_prior × parent_boost_factor.
        parent_boost_factor derived from parent's confidence and
        the causal strength encoded in the template relationship.
        """
```

---

## Module 5: `evidence/` — Evidence Management

### `matcher.py`

```python
from dataclasses import dataclass, field
from typing import Optional

from ..graph.nodes import ObservationNode, HypothesisNode
from ..graph.graph import ReasoningGraph
from ..graph.edges import EvidentialEdge
from .template_loader import TemplateLoader, HypothesisTemplate


@dataclass
class MatchResult:
    """Outcome of matching one evidence node against templates."""
    evidence_id:   str
    matched:       bool
    hypothesis_ids: list[str] = field(default_factory=list)
    edge_ids:      list[str]  = field(default_factory=list)
    unmatched:     bool       = False  # True = pass to abductive engine


class EvidenceMatcher:
    """
    For each incoming ObservationNode, determines which
    hypothesis templates it satisfies (fully or partially)
    and creates EvidentialEdges with calibrated likelihood ratios.

    Matching is evaluated against each template's
    activation_conditions using a rule interpreter.
    """

    def __init__(self,
                 graph: ReasoningGraph,
                 template_loader: TemplateLoader):
        self.graph = graph
        self.template_loader = template_loader

    def match(self, evidence: ObservationNode) -> MatchResult:
        """
        Match one evidence node against all applicable templates.
        For each match:
          - Activate the corresponding dormant HypothesisNode
          - Create an EvidentialEdge with the correct likelihood ratio
        Returns a MatchResult with match status and created IDs.
        """

    def match_batch(self,
                     evidence_list: list[ObservationNode]) -> list[MatchResult]:
        """Match a batch of evidence nodes. Returns results in order."""

    def _evaluate_conditions(self,
                              evidence: ObservationNode,
                              template: HypothesisTemplate) -> bool:
        """
        Evaluate whether evidence satisfies all of a template's
        activation_conditions. Returns True if all pass.
        """

    def _calibrate_edge(self,
                         evidence: ObservationNode,
                         template: HypothesisTemplate) -> tuple[float, float]:
        """
        Look up P(E|H) and P(E|¬H) from the template's
        likelihood_ratios for this evidence type.
        Falls back to default values if not specified.
        """
```

---

### `requester.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..graph.nodes import HypothesisNode


class EvidenceRequestPriority(Enum):
    CRITICAL = "critical"   # blocks hypothesis resolution
    HIGH     = "high"       # would significantly change confidence
    MEDIUM   = "medium"     # useful but not blocking
    LOW      = "low"        # nice to have


@dataclass
class EvidenceRequest:
    """
    A targeted request for a specific piece of evidence,
    emitted by an INCONCLUSIVE hypothesis that needs more
    information to resolve.

    This is what makes Holmes active rather than passive —
    the graph tells the collectors what to look for next.
    """
    id:               str       = field(default_factory=lambda: str(uuid.uuid4()))
    requesting_hyp_id: str      = ""
    evidence_type:    str       = ""   # what kind of evidence is needed
    subject:          str       = ""   # which feature / segment / model layer
    collector_id:     str       = ""   # which collector should run
    collector_params: dict      = field(default_factory=dict)
    priority:         EvidenceRequestPriority = EvidenceRequestPriority.HIGH
    rationale:        str       = ""   # why this evidence is needed
    expected_impact:  float     = 0.0  # how much it would reduce uncertainty
    created_at:       datetime  = field(default_factory=datetime.utcnow)
    fulfilled_at:     Optional[datetime] = None
    fulfilled_by:     Optional[str]      = None  # evidence node ID


class EvidenceRequester:
    """
    Analyzes INCONCLUSIVE hypotheses and computes the
    evidence request that would most reduce uncertainty —
    the highest-information-gain next measurement.

    Uses the expected information gain:
      IG(E, H) = H(H) - E[H(H|E)]
    where H is entropy and E is expectation over possible
    evidence outcomes.

    Emits requests to a queue consumed by evidence collectors.
    """

    def __init__(self, graph, request_queue):
        self.graph = graph
        self.request_queue = request_queue

    def scan_and_emit(self) -> list[EvidenceRequest]:
        """
        Scan all INCONCLUSIVE hypotheses.
        For each, compute the optimal next evidence request.
        Emit requests to the queue.
        Return all emitted requests.
        """

    def compute_best_request(self,
                              hyp: HypothesisNode) -> Optional[EvidenceRequest]:
        """
        For a given INCONCLUSIVE hypothesis, determine the
        single evidence request that would most reduce
        uncertainty (maximize information gain).
        """

    def _estimate_information_gain(self,
                                    hyp: HypothesisNode,
                                    evidence_type: str) -> float:
        """
        Estimate how much entropy reduction we'd get from
        receiving evidence of this type for this hypothesis.
        Uses current confidence and template likelihood ratios.
        """
```

---

## Module 6: `query/` — Graph Querying & Export

### `tracer.py`

```python
from dataclasses import dataclass, field
from typing import Optional

from ..graph.graph import ReasoningGraph
from ..graph.nodes import BaseNode
from ..graph.edges import EdgeType


@dataclass
class ReasoningTrace:
    """
    The full causal path that explains how the graph reached
    a conclusion. This is the auditability artifact —
    what makes Holmes trustworthy rather than a black box.
    """
    finding_id:      str
    hypothesis_id:   str
    path:            list[dict]   # ordered list of nodes and edges traversed
    # [{"node_id": ..., "node_type": ..., "confidence": ..., "contribution": ...}]
    narrative:       str          # human-readable explanation of the path
    total_confidence: float
    evidence_summary: list[dict]
    # [{"evidence_id": ..., "type": ..., "weight": ..., "polarity": ...}]


class CausalPathTracer:
    """
    Traces the causal path from root cause to observed symptom
    through the reasoning graph.

    Uses a modified Dijkstra's algorithm on causal edges,
    weighted by confidence × edge strength.

    The path is the explanation narrative: it shows exactly
    which evidence activated which hypotheses, which hypotheses
    caused others, and how confidence accumulated.
    """

    def __init__(self, graph: ReasoningGraph):
        self.graph = graph

    def trace(self, hypothesis_id: str) -> ReasoningTrace:
        """
        Trace the full reasoning path for a given hypothesis.
        Walks backward from the hypothesis through its evidence
        and causal parents to build the complete explanation.
        """

    def find_root_causes(self, hypothesis_id: str) -> list[str]:
        """
        Find the deepest causal ancestors of a hypothesis —
        the nodes with no incoming causal edges that are
        still in the causal chain. These are the root causes.
        """

    def find_shortest_causal_path(self,
                                   source_id: str,
                                   target_id: str) -> Optional[list[str]]:
        """
        Find the shortest causal path between two nodes.
        Returns ordered list of node IDs, or None if no path.
        """

    def generate_narrative(self, trace: ReasoningTrace) -> str:
        """
        Convert a ReasoningTrace into a human-readable narrative.
        Example output:
        "We detected that feature 'days_since_purchase' has a
         suspiciously high correlation with the target (E1, confidence 0.9).
         This activated the TEMPORAL_LEAKAGE hypothesis. Point-in-time
         simulation (E2) confirmed the feature uses post-event data,
         pushing confidence to 0.91. This caused MODEL_OVERFIT_VALIDATION
         (0.78 confidence) via the known causal relationship."
        """
```

---

### `ranker.py`

```python
from dataclasses import dataclass, field
from typing import Optional

from ..graph.graph import ReasoningGraph
from ..graph.nodes import HypothesisNode


@dataclass
class RankedHypothesis:
    """A hypothesis with its composite ranking score."""
    hypothesis_id:    str
    hypothesis_type:  str
    claim:            str
    confidence:       float
    business_impact:  float
    actionability:    float   # how directly actionable is the recommendation
    centrality:       float   # how central is this node in the graph
    composite_score:  float   # final ranking score
    rank:             int


class HypothesisRanker:
    """
    Ranks all confirmed hypotheses by a composite score
    for presentation and finding generation priority.

    Composite score = w1×confidence + w2×business_impact
                    + w3×actionability + w4×centrality

    Weights are configurable and can be tuned to the
    user's priorities (e.g., emphasize business_impact
    for an executive report, actionability for engineers).
    """

    def __init__(self,
                 confidence_weight:     float = 0.35,
                 business_impact_weight: float = 0.30,
                 actionability_weight:  float = 0.20,
                 centrality_weight:     float = 0.15):
        self.weights = {
            "confidence":      confidence_weight,
            "business_impact": business_impact_weight,
            "actionability":   actionability_weight,
            "centrality":      centrality_weight,
        }

    def rank(self,
             graph: ReasoningGraph,
             top_k: Optional[int] = None) -> list[RankedHypothesis]:
        """
        Rank all confirmed hypotheses in the graph.
        Returns sorted list, best first.
        Optionally limited to top_k results.
        """

    def _compute_centrality(self,
                             hyp: HypothesisNode,
                             graph: ReasoningGraph) -> float:
        """
        Compute a PageRank-like centrality score for this
        hypothesis in the current graph.
        High centrality = this hypothesis is connected to many
        others and is a hub in the reasoning structure.
        """

    def _compute_actionability(self, hyp: HypothesisNode) -> float:
        """
        Score how directly actionable this hypothesis is.
        Hypotheses with well-defined remediation steps
        and lower estimated effort score higher.
        """
```

---

## Module 7: `store/` — Persistence

### `event_log.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    NODE_ADDED         = "node_added"
    NODE_REMOVED       = "node_removed"
    NODE_STATUS_CHANGED = "node_status_changed"
    EDGE_ADDED         = "edge_added"
    EDGE_WEIGHT_UPDATED = "edge_weight_updated"
    CONFIDENCE_UPDATED = "confidence_updated"
    PROPAGATION_RUN    = "propagation_run"
    CONFLICT_DETECTED  = "conflict_detected"
    EVIDENCE_REQUESTED = "evidence_requested"
    HYPOTHESIS_SPAWNED = "hypothesis_spawned"
    FINDING_GENERATED  = "finding_generated"


@dataclass
class GraphEvent:
    """
    Every change to the reasoning graph is stored as an
    immutable event. The graph is event-sourced — its
    state at any point in time can be reconstructed by
    replaying events up to that timestamp.
    """
    id:            str       = field(default_factory=lambda: str(uuid.uuid4()))
    graph_id:      str       = ""
    event_type:    EventType = None
    payload:       dict      = field(default_factory=dict)
    occurred_at:   datetime  = field(default_factory=datetime.utcnow)
    triggered_by:  str       = ""   # evidence ID, user ID, or system process


class GraphEventLog:
    """
    Append-only log of all graph changes.
    Enables:
    - Full replay of any investigation from scratch
    - Diff between any two points in time
    - Audit trail for regulatory or compliance needs
    - Training data for meta-learning across investigations
    """

    def __init__(self, storage_backend):
        self.storage = storage_backend

    def append(self, event: GraphEvent) -> None:
        """Append an event to the log. Immutable — no updates."""

    def replay(self, graph_id: str,
               up_to: Optional[datetime] = None) -> "ReasoningGraph":
        """
        Reconstruct the graph state by replaying all events
        for graph_id, optionally up to a specific timestamp.
        """

    def events_since(self, graph_id: str,
                     since: datetime) -> list[GraphEvent]:
        """Return all events for a graph after a given timestamp."""

    def diff(self, graph_id: str,
             from_time: datetime,
             to_time: datetime) -> list[GraphEvent]:
        """Return events between two timestamps (the graph diff)."""
```

---

## Module Dependency Map

```
store/event_log          ← used by all modules for audit
store/graph_store        ← persists graph snapshots

graph/nodes, edges       ← foundational types, no dependencies
graph/graph              ← depends on nodes, edges

evidence/matcher         ← depends on graph, construction/template_loader
evidence/requester       ← depends on graph/nodes
hypothesis/lifecycle     ← depends on graph, store/event_log
hypothesis/spawner       ← depends on graph, construction/template_loader

inference/decay          ← depends on graph/nodes, graph/edges
inference/conflict       ← depends on graph, inference/propagator
inference/propagator     ← depends on graph (core algorithm)

construction/template_loader  ← standalone (loads from config)
construction/abductive        ← depends on graph/nodes, LLM client
construction/builder          ← depends on all of the above

query/tracer             ← depends on graph
query/ranker             ← depends on graph
query/exporter           ← depends on graph, query/tracer, query/ranker
```
