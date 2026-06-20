# Holmes SDK — Session Module: Full Design

The `Session` is the spine of the SDK — the single stateful entry point a
data scientist holds in their notebook. Every other capability (investigation,
evidence collection, graph interaction, conversation, export) is reached
through it. This document specifies the modules, classes, methods, and
attributes needed to implement it.

---

## Module Layout

```
holmes/
└── session/
    ├── session.py          # The Session class itself — the spine
    ├── config.py           # SessionConfig, scope & threshold settings
    ├── investigation.py     # InvestigationRunner, Plan, progress streaming
    ├── collection.py       # Evidence-only collection interface
    ├── graph_handle.py     # GraphHandle — the session's view into the ReasoningGraph
    ├── conversation.py     # Conversational follow-up (.ask on session/finding)
    ├── findings_view.py    # FindingsCollection — query/filter/export findings
    ├── report.py           # Report generation (pdf, slides, dataframe export)
    ├── comparison.py       # holmes.compare(session_a, session_b)
    ├── knowledge.py        # session.tell() — domain knowledge injection
    └── display/
        ├── repr_html.py     # _repr_html_ implementations for notebook display
        └── progress.py      # Live progress bar / streaming display widget
```

---

## Module 1: `session.py` — The Session Class

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, Callable
import uuid

from .config import SessionConfig
from .investigation import InvestigationRunner, InvestigationPlan, InvestigationHandle
from .collection import EvidenceCollectionInterface
from .graph_handle import GraphHandle
from .conversation import ConversationInterface
from .findings_view import FindingsCollection
from .report import ReportGenerator
from .knowledge import KnowledgeInjector


class Session:
    """
    The central entry point for a Holmes investigation.

    Holds the reasoning graph, evidence store, and findings for one
    investigation context (one dataset + optional model). Every other
    capability — running investigations, collecting evidence, querying
    the graph, conversing, exporting — is reached as a method or
    attribute on this object.

    Designed to be created once per notebook cell-flow and reused
    across many subsequent calls, mirroring how a notebook kernel
    itself behaves.

    Example:
        session = holmes.investigate(dataset, model=my_model)
        session.investigate()
        session.graph.hypotheses(status="active")
        session.findings[0].explain()
    """

    # ---- Identity & core state ----
    id:               str
    created_at:       datetime
    dataset_ref:      "DatasetBinding"        # from MEMS
    model_ref:        Optional["ModelBinding"] # from MEMS, optional
    config:           SessionConfig

    # ---- Core subsystems (composition, not inheritance) ----
    graph:            GraphHandle              # the reasoning graph interface
    findings:         FindingsCollection        # query/filter/export findings
    evidence_store:   "EvidenceStore"           # raw evidence, independent of graph

    # ---- Internal collaborators ----
    _investigation_runner: InvestigationRunner
    _collection_interface: EvidenceCollectionInterface
    _conversation:         ConversationInterface
    _report_generator:     ReportGenerator
    _knowledge_injector:    KnowledgeInjector

    # ---- Lifecycle / bookkeeping ----
    status:           str   # "idle" | "investigating" | "ready" | "error"
    last_investigation_at: Optional[datetime]
    investigation_history: list["InvestigationHandle"]
    tags:             dict   # free-form user metadata, e.g. {"env": "staging"}

    def __init__(self,
                 dataset_ref,
                 model_ref=None,
                 config: Optional[SessionConfig] = None):
        """
        Construct a new session. Initializes an empty ReasoningGraph
        scoped to this dataset/model pair, sets up the evidence store,
        and wires all internal collaborators.

        Typically not called directly — use holmes.investigate(...)
        as the public factory function.
        """

    # ---------------------------------------------------------------
    # INVESTIGATION
    # ---------------------------------------------------------------

    def investigate(self,
                     scope: Optional[Union[str, list[str]]] = None,
                     lazy: bool = False,
                     min_severity: str = "medium",
                     stream: bool = True,
                     timeout: Optional[int] = None
                     ) -> Union["InvestigationHandle", "InvestigationPlan"]:
        """
        Run an investigation. The single entry point for both
        "general" and "individual" investigations — scope determines
        breadth.

        Args:
            scope: None = run all subsystems ("general investigation").
                   A string like "missing_values" or "leakage" = run
                   just that subsystem ("individual investigation").
                   A list of strings = run a custom subset.
            lazy: if True, returns an InvestigationPlan describing what
                  would run and its cost estimate, without executing it.
            min_severity: findings below this severity are still stored
                  and queryable, but not surfaced in the default output.
            stream: if True (default, notebook-friendly), shows a live
                  progress display as subsystems complete and yields
                  partial results incrementally.
            timeout: optional max seconds before returning partial results.

        Returns:
            InvestigationHandle if lazy=False (run has started/completed).
            InvestigationPlan if lazy=True (nothing has run yet).
        """

    def stop_investigation(self) -> None:
        """
        Interrupt a currently running investigation. Already-completed
        subsystem results remain in the graph; in-flight work is
        cancelled cooperatively at the next safe checkpoint.
        """

    @property
    def is_investigating(self) -> bool:
        """True while an investigation is actively running."""

    # ---------------------------------------------------------------
    # EVIDENCE COLLECTION (no investigation / hypothesis generation)
    # ---------------------------------------------------------------

    def collect(self,
                evidence_type: Union[str, list[str]],
                link_to_graph: bool = True,
                **collector_params) -> "EvidenceBatch":
        """
        Collect raw evidence without running a full investigation or
        triggering hypothesis generation eagerly.

        Args:
            evidence_type: which collector(s) to run, e.g.
                "missing_values", "duplicates", or a list of several.
            link_to_graph: if True (default), collected evidence is
                still added to the reasoning graph in the background
                (matched against templates, may activate dormant
                hypotheses) — it just doesn't force generation/synthesis
                of a Finding. If False, evidence is fully standalone.
            **collector_params: passed through to the underlying
                collector (e.g. columns=[...], sample_size=...).

        Returns:
            EvidenceBatch — the raw evidence results, inspectable
            directly (e.g. as a DataFrame) regardless of link_to_graph.
        """

    # ---------------------------------------------------------------
    # CONVERSATIONAL INTERFACE
    # ---------------------------------------------------------------

    def ask(self, question: str) -> "ConversationResponse":
        """
        Ask a free-form question about this investigation. Routed
        through the ConversationInterface, which has access to the
        full graph, evidence store, and findings as context.

        Supports what-if framing ("what if the leakage in X were
        fixed?"), which is dispatched to graph.simulate() under the hood.

        Example:
            session.ask("why is the elderly segment underperforming?")
            session.ask("what if I removed feature hospital_id?")
        """

    # ---------------------------------------------------------------
    # DOMAIN KNOWLEDGE INJECTION
    # ---------------------------------------------------------------

    def tell(self,
             statement: str,
             effective_date: Optional[datetime] = None,
             confidence: float = 0.9) -> "ContextNode":
        """
        Inject domain knowledge as a low-friction one-liner. Parsed
        (via KnowledgeInjector) into a ContextNode or seed
        HypothesisNode in the reasoning graph.

        Example:
            session.tell("we changed the churn label definition on 2024-03-01")
        """

    # ---------------------------------------------------------------
    # EXPORT / REPORTING
    # ---------------------------------------------------------------

    @property
    def report(self) -> "ReportGenerator":
        """
        Access point for report generation:
            session.report.to_pdf()
            session.report.to_slides()
            session.report.executive_summary()
        """

    def to_dataframe(self, what: str = "findings") -> "pandas.DataFrame":
        """
        Convenience export. what: "findings" | "hypotheses" | "evidence".
        Equivalent to session.findings.to_dataframe() for the default case.
        """

    # ---------------------------------------------------------------
    # STATE / PERSISTENCE
    # ---------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Persist the full session state (graph, evidence, findings,
        config) to disk or the configured backend. Returns the path
        or session ID it was saved under.
        """

    @classmethod
    def load(cls, path_or_id: str) -> "Session":
        """Restore a previously saved session."""

    def fork(self) -> "Session":
        """
        Create a copy of this session (graph + evidence) that can be
        modified independently — e.g. to try a what-if scenario
        destructively without affecting the original.
        """

    # ---------------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Rich notebook display: dataset/model summary, investigation
        status, top findings preview, graph health snapshot.
        """

    def summary(self) -> "SessionSummary":
        """Structured, non-HTML summary object — useful outside notebooks."""
```

---

## Module 2: `config.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionConfig:
    """
    User-tunable settings for how a session behaves. Passed at
    construction or modifiable via session.config.update(...).
    """
    default_min_severity:     str   = "medium"   # filters default investigate() output
    default_stream:           bool  = True
    confirmation_threshold_overrides: dict = field(default_factory=dict)
    # hypothesis_type -> override confirmation threshold

    auto_link_evidence_to_graph: bool = True  # default for session.collect()
    max_parallel_subsystems:  int   = 4
    investigation_timeout_s:  Optional[int] = None

    llm_model_for_conversation: str = "claude-sonnet-4-6"
    abductive_novelty_discount: float = 0.2

    storage_backend:          str   = "local"   # "local" | "remote" | "in_memory"
    persist_on_exit:          bool  = False

    def update(self, **kwargs) -> "SessionConfig":
        """Update config fields in place, validating types/ranges."""

    def for_context(self, dataset_type: str, domain: str) -> "SessionConfig":
        """
        Return a config tuned for a known dataset/domain combination
        (e.g. tighter thresholds for finance, looser for exploratory
        research datasets).
        """
```

---

## Module 3: `investigation.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable


class InvestigationStatus(Enum):
    PENDING     = "pending"
    RUNNING     = "running"
    COMPLETED   = "completed"
    PARTIAL     = "partial"      # stopped early / timed out
    FAILED      = "failed"


@dataclass
class SubsystemCostEstimate:
    """Cost estimate for a single subsystem within a plan."""
    subsystem_name:    str
    estimated_seconds: float
    requires_surrogate_model: bool = False
    requires_production_sample: bool = False
    notes:             str = ""


@dataclass
class InvestigationPlan:
    """
    Returned when investigate(lazy=True) is called. Describes what
    WOULD run, without running it — lets the user inspect cost
    before committing, similar to Spark's lazy execution plans.
    """
    session_id:          str
    scope:                list[str]            # subsystems that would run
    subsystem_estimates:  list[SubsystemCostEstimate]
    estimated_total_seconds: float
    estimated_evidence_count: int
    warnings:             list[str] = field(default_factory=list)
    # e.g. ["No production sample provided — coverage analysis will be skipped"]

    def run(self, stream: bool = True) -> "InvestigationHandle":
        """Execute the plan for real."""

    def explain(self) -> str:
        """Human-readable breakdown of what will run and why."""


@dataclass
class InvestigationHandle:
    """
    Returned when investigate() runs (lazy=False). A live or completed
    handle representing one investigation run. Supports streaming
    inspection while running, and full results once done.
    """
    id:               str
    session_id:        str
    scope:             list[str]
    status:            InvestigationStatus
    started_at:        datetime
    completed_at:      Optional[datetime]
    progress:          "ProgressState"     # see display/progress.py

    new_evidence_ids:      list[str] = field(default_factory=list)
    new_hypothesis_ids:    list[str] = field(default_factory=list)
    new_finding_ids:       list[str] = field(default_factory=list)

    def wait(self, timeout: Optional[int] = None) -> "InvestigationHandle":
        """Block until completion (or timeout); useful outside notebooks."""

    def top_findings(self, k: int = 5) -> list["Finding"]:
        """Convenience accessor — the triage-first view of this run's output."""

    def on_subsystem_complete(self, callback: Callable) -> None:
        """Register a callback fired each time a subsystem finishes."""

    def _repr_html_(self) -> str:
        """Live-updating progress display while running; summary once done."""


class InvestigationRunner:
    """
    Internal orchestrator. Not typically used directly by end users —
    Session.investigate() delegates to this.

    Responsible for: resolving scope into a concrete subsystem list,
    estimating cost (for lazy plans), executing subsystems with the
    configured parallelism, streaming partial results back to the
    InvestigationHandle, and feeding all outputs into the session's
    GraphHandle / FindingsCollection.
    """

    def __init__(self, session: "Session"):
        self.session = session

    def build_plan(self, scope: Optional[list[str]],
                    min_severity: str) -> InvestigationPlan:
        """Resolve scope and produce a cost-estimated plan, without running."""

    def execute(self, plan: InvestigationPlan,
                stream: bool, timeout: Optional[int]) -> InvestigationHandle:
        """Run the plan, streaming progress, returning a live handle."""

    def _resolve_scope(self, scope) -> list[str]:
        """
        None -> all registered subsystems.
        str -> [that subsystem].
        list -> validated subset.
        Raises a clear error for unknown subsystem names, suggesting
        close matches (e.g. "leakge" -> did you mean "leakage"?).
        """

    def _run_subsystem(self, subsystem_name: str) -> "SubsystemResult":
        """Invoke one subsystem, capture evidence, feed into graph builder."""
```

---

## Module 4: `collection.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EvidenceBatch:
    """
    Result of session.collect(...). Raw, inspectable evidence —
    independent of whether it was linked into the reasoning graph.
    """
    id:               str
    evidence_type:     str
    collected_at:      datetime
    items:             list["ObservationNode"]
    linked_to_graph:   bool
    activated_hypothesis_ids: list[str] = field(default_factory=list)
    # populated only if linked_to_graph=True and matches occurred

    def to_dataframe(self) -> "pandas.DataFrame":
        """Flatten evidence items into a tabular view for direct inspection."""

    def filter(self, **kwargs) -> "EvidenceBatch":
        """e.g. batch.filter(subject="income") — narrow down items."""

    def link_now(self) -> list[str]:
        """
        If collected with link_to_graph=False, link this batch into
        the graph now. Returns any newly activated hypothesis IDs.
        """

    def _repr_html_(self) -> str:
        """Compact table view of collected evidence items."""


class EvidenceCollectionInterface:
    """
    Internal collaborator behind Session.collect(). Wraps the
    individual evidence collectors (one per subsystem-evidence-type)
    and standardizes their output into EvidenceBatch objects.
    """

    def __init__(self, session: "Session"):
        self.session = session
        self._registry: dict[str, "BaseCollector"] = {}

    def collect(self, evidence_type, link_to_graph: bool,
                **params) -> EvidenceBatch:
        """Dispatch to the correct collector(s), assemble EvidenceBatch."""

    def available_types(self) -> list[str]:
        """List all registered evidence_type collector names."""

    def register_collector(self, evidence_type: str,
                           collector: "BaseCollector") -> None:
        """Allow custom/user-defined collectors to be plugged in."""
```

---

## Module 5: `graph_handle.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Union

from ..reasoning_graph.graph.graph import ReasoningGraph
from ..reasoning_graph.graph.nodes import HypothesisNode, NodeStatus
from ..reasoning_graph.query.tracer import ReasoningTrace
from ..reasoning_graph.inference.propagator import PropagationResult


@dataclass
class SimulationResult:
    """Result of a what-if simulation run via graph.simulate()."""
    intervention_description: str
    affected_hypothesis_ids:  list[str]
    confidence_deltas:        dict[str, float]   # hypothesis_id -> delta
    projected_metric_change:  Optional[dict] = None
    # e.g. {"auc": +0.04, "false_positive_rate": -0.02}
    narrative:                 str = ""

    def _repr_html_(self) -> str:
        """Before/after comparison view."""


class GraphHandle:
    """
    The session's user-facing interface into its ReasoningGraph.
    Wraps the lower-level graph module (nodes/edges/inference) with
    an ergonomic, notebook-friendly API surface — this is what
    "session.graph.X" resolves to.
    """

    def __init__(self, session: "Session", graph: ReasoningGraph):
        self.session = session
        self._graph = graph

    # ---- Query ----

    def hypotheses(self,
                   status: Optional[Union[str, list[str]]] = None,
                   min_confidence: Optional[float] = None,
                   hypothesis_type: Optional[str] = None,
                   subject: Optional[str] = None) -> list[HypothesisNode]:
        """Filtered query over hypothesis nodes in the graph."""

    def evidence_for(self, hypothesis_id: str) -> list["ObservationNode"]:
        """All evidence nodes connected (supporting or contradicting) to a hypothesis."""

    def evidence(self,
                evidence_type: Optional[str] = None,
                subject: Optional[str] = None) -> list["ObservationNode"]:
        """Filtered query over all evidence nodes, independent of hypotheses."""

    def get(self, node_id: str) -> Optional["BaseNode"]:
        """Direct node lookup by ID."""

    # ---- Traverse / Explain ----

    def trace(self, finding_or_hypothesis_id: str) -> ReasoningTrace:
        """The reasoning trace — full "why" behind a conclusion."""

    def root_causes(self, hypothesis_id: str) -> list[HypothesisNode]:
        """Causal ancestors with no further upstream cause in the graph."""

    def related(self, hypothesis_id: str,
                edge_type: Optional[str] = None) -> list[HypothesisNode]:
        """Neighboring hypotheses (causal, competitive, corroborating)."""

    # ---- Simulate ----

    def simulate(self, description: str,
                 **intervention_params) -> SimulationResult:
        """
        Run a what-if intervention against the graph. Creates an
        InterventionNode, dispatches to ISS for counterfactual
        simulation, and returns the projected effect on hypotheses
        and (if available) downstream metrics.

        Example:
            session.graph.simulate("remove feature X")
            session.graph.simulate(remove_feature="hospital_id")
        """

    def what_if(self, **kwargs) -> SimulationResult:
        """Alias for simulate() with keyword-only convenience syntax."""

    # ---- Override / Correct ----

    def reject(self, hypothesis_id: str, reason: str) -> None:
        """
        User override: mark a hypothesis as rejected despite its
        current confidence. Feeds into the belief calibration store
        so future investigations weight similar evidence differently.
        """

    def boost(self, hypothesis_id: str, reason: str,
              target_confidence: Optional[float] = None) -> None:
        """
        User override: confirm/boost a hypothesis based on external
        verification (e.g. a domain expert manually checked it).
        """

    def merge(self, hypothesis_id_a: str, hypothesis_id_b: str) -> str:
        """Manually merge two hypotheses the user believes are equivalent."""

    # ---- Inspect graph state ----

    @property
    def stats(self) -> dict:
        """
        Quick health snapshot: node counts by type/status, edge
        counts by type, last propagation run timestamp, number of
        inconclusive hypotheses awaiting evidence.
        """

    def pending_evidence_requests(self) -> list["EvidenceRequest"]:
        """Active requests the graph has emitted for more evidence."""

    def visualize(self, focus: Optional[str] = None) -> "GraphVisualization":
        """
        Render the graph (or a subgraph around `focus`) as an
        interactive notebook widget.
        """

    def _repr_html_(self) -> str:
        """Compact graph summary: node/edge counts, top active hypotheses."""
```

---

## Module 6: `conversation.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ConversationResponse:
    """Result of session.ask(...) or finding.ask(...)."""
    question:           str
    answer:             str
    referenced_node_ids: list[str] = field(default_factory=list)
    triggered_simulation: Optional["SimulationResult"] = None
    confidence:          Optional[float] = None
    follow_up_suggestions: list[str] = field(default_factory=list)

    def _repr_html_(self) -> str:
        """Chat-style rendering of the answer with linked node references."""

    def show_trace(self) -> "ReasoningTrace":
        """If the answer drew on specific graph reasoning, show it."""


@dataclass
class ConversationTurn:
    """One turn in a session's conversation history."""
    question:    str
    response:    ConversationResponse
    asked_at:    datetime
    scope:       str   # "session" | finding_id


class ConversationInterface:
    """
    Backs Session.ask() and Finding.ask(). Maintains conversation
    history per scope and routes questions to either:
      - a direct graph query (if the question matches a known pattern)
      - a what-if simulation (if framed as "what if...")
      - an LLM call with full graph/evidence/findings context (general case)
    """

    def __init__(self, session: "Session"):
        self.session = session
        self.history: list[ConversationTurn] = []

    def ask(self, question: str,
            scope: Optional[str] = None) -> ConversationResponse:
        """Main entry point. Classifies and routes the question, returns response."""

    def _is_what_if(self, question: str) -> bool:
        """Lightweight classifier: does this question imply a simulation?"""

    def _build_context_payload(self, scope: Optional[str]) -> dict:
        """
        Assemble the graph state, relevant findings, and evidence
        summaries to ground the LLM's answer in this session's
        actual data — not just general ML knowledge.
        """

    def _route_to_simulation(self, question: str) -> ConversationResponse:
        """Parse intervention parameters from the question, call graph.simulate()."""

    def conversation_history(self, scope: Optional[str] = None) -> list[ConversationTurn]:
        """Retrieve past turns, optionally filtered to one finding's scope."""
```

---

## Module 7: `findings_view.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Union, Iterator


class FindingsCollection:
    """
    The queryable, filterable, exportable collection of findings
    produced by a session. session.findings behaves like a smart
    list — indexable, iterable, and chainable with filters.
    """

    def __init__(self, session: "Session"):
        self.session = session
        self._items: list["Finding"] = []

    def __getitem__(self, index: int) -> "Finding":
        """session.findings[0] — indexed access, ranked order."""

    def __iter__(self) -> Iterator["Finding"]:
        """Iterate findings in ranked order."""

    def __len__(self) -> int:
        ...

    def filter(self,
               severity: Optional[Union[str, list[str]]] = None,
               finding_type: Optional[str] = None,
               min_confidence: Optional[float] = None,
               subject: Optional[str] = None) -> "FindingsCollection":
        """Return a new filtered FindingsCollection (chainable)."""

    def sort_by(self, key: str = "composite_score",
                descending: bool = True) -> "FindingsCollection":
        """Re-sort findings by confidence, severity, business_impact, etc."""

    def top(self, k: int = 5) -> list["Finding"]:
        """Top-k by current ranking — the triage view."""

    def to_dataframe(self) -> "pandas.DataFrame":
        """Flatten findings into a tabular view."""

    def to_dict_list(self) -> list[dict]:
        """JSON-serializable representation."""

    def by_id(self, finding_id: str) -> Optional["Finding"]:
        """Direct lookup."""

    def _repr_html_(self) -> str:
        """Card-style summary list, ranked, with severity badges."""


@dataclass
class Finding:
    """
    A single finding — wraps the FindingNode from the reasoning graph
    with session-aware convenience methods.
    """
    id:                str
    finding_type:       str
    severity:           str
    confidence:          float
    claim:               str
    subject:             str
    recommendations:     list[str]
    estimated_fix_impact: Optional[float]
    _session:            "Session" = field(repr=False, default=None)

    def explain(self) -> "ReasoningTrace":
        """Shortcut for session.graph.trace(self.id)."""

    def ask(self, question: str) -> ConversationResponse:
        """Conversational follow-up scoped to this specific finding."""

    def simulate_fix(self) -> "SimulationResult":
        """Run the recommended fix as a what-if simulation."""

    def accept(self, note: Optional[str] = None) -> None:
        """User marks this finding as valid/useful — feeds calibration."""

    def dismiss(self, reason: str) -> None:
        """User marks this finding as irrelevant/incorrect — feeds calibration."""

    def to_slide(self) -> "Slide":
        """Export this single finding as a slide object for a review deck."""

    def _repr_html_(self) -> str:
        """Collapsed card by default; expands to full trace on .explain()."""
```

---

## Module 8: `report.py`

```python
from dataclasses import dataclass
from typing import Optional


class ReportGenerator:
    """
    Backs session.report. Produces shareable artifacts from the
    session's findings, graph, and evidence.
    """

    def __init__(self, session: "Session"):
        self.session = session

    def executive_summary(self) -> str:
        """Short natural-language summary of the investigation's key findings."""

    def to_pdf(self, path: Optional[str] = None,
               include_trace: bool = False) -> str:
        """Full report as a PDF. Returns the file path."""

    def to_slides(self, path: Optional[str] = None,
                  max_findings: int = 5) -> str:
        """Generate a slide deck (pptx) covering top findings. Returns file path."""

    def to_markdown(self) -> str:
        """Markdown version of the report, for embedding in docs/wikis."""

    def to_html(self) -> str:
        """Standalone HTML report."""

    def schedule(self, cadence: str, destination: str) -> "ScheduledReport":
        """
        Configure a recurring report (e.g. "weekly" health summary sent
        to a Slack channel or email). Useful for production sessions.
        """
```

---

## Module 9: `comparison.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfidenceShift:
    hypothesis_type:   str
    subject:           str
    confidence_before: float
    confidence_after:  float
    delta:             float


@dataclass
class SessionDiff:
    """Result of holmes.compare(session_a, session_b)."""
    session_a_id:       str
    session_b_id:       str
    new_findings:        list["Finding"]
    resolved_findings:   list["Finding"]
    persisted_findings:  list["Finding"]
    confidence_shifts:   list[ConfidenceShift]
    overall_health_delta: Optional[float] = None

    def summary(self) -> str:
        """Human-readable narrative of what changed between sessions."""

    def _repr_html_(self) -> str:
        """Side-by-side / delta view."""


def compare(session_a: "Session", session_b: "Session") -> SessionDiff:
    """
    Module-level function: holmes.compare(v1, v2).
    Diffs two sessions' findings and hypothesis confidences,
    typically used across model versions or data refresh cycles.
    """
```

---

## Module 10: `knowledge.py`

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ParsedKnowledge:
    """Intermediate structure after parsing a session.tell() statement."""
    statement:        str
    node_type:         str   # "context" | "hypothesis_seed"
    context_type:      Optional[str] = None
    affected_hypothesis_types: list[str] = field(default_factory=list)
    effective_date:    Optional[datetime] = None
    confidence:        float = 0.9


class KnowledgeInjector:
    """
    Backs Session.tell(). Parses free-text domain knowledge into
    graph-ready ContextNode or seed HypothesisNode objects via an
    LLM parsing step, then wires them into the graph.
    """

    def __init__(self, session: "Session", llm_client):
        self.session = session
        self.llm_client = llm_client

    def inject(self, statement: str,
               effective_date: Optional[datetime],
               confidence: float) -> "ContextNode":
        """Main entry point: parse and wire the statement into the graph."""

    def _parse(self, statement: str) -> ParsedKnowledge:
        """LLM call: classify and structure the free-text statement."""

    def _wire_into_graph(self, parsed: ParsedKnowledge) -> "ContextNode":
        """Create the node and contextual/seed edges from parsed knowledge."""
```

---

## Module 11: `display/`

### `progress.py`

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgressState:
    """
    Live state for a running InvestigationHandle, rendered as a
    notebook progress widget (ipywidgets-based) or a plain text
    fallback outside Jupyter.
    """
    total_subsystems:     int
    completed_subsystems: int
    current_subsystem:    Optional[str]
    subsystem_statuses:   dict[str, str]   # name -> "pending"|"running"|"done"|"failed"
    elapsed_seconds:      float
    estimated_remaining_seconds: Optional[float]

    def render(self) -> "IPython.display.DisplayObject":
        """Render/update the live progress widget."""

    def as_text(self) -> str:
        """Plain-text fallback for non-notebook environments."""
```

### `repr_html.py`

```python
class ReprHtmlMixin:
    """
    Shared HTML rendering helpers used by Session, Finding,
    GraphHandle, EvidenceBatch, etc. so every Holmes object has a
    consistent visual language in the notebook (badges for severity,
    confidence bars, collapsible sections for traces).
    """

    def _confidence_bar_html(self, confidence: float) -> str: ...
    def _severity_badge_html(self, severity: str) -> str: ...
    def _collapsible_section_html(self, title: str, content_html: str) -> str: ...
```

---

## Top-Level Factory Function

```python
# holmes/__init__.py

def investigate(dataset, model=None, config: Optional["SessionConfig"] = None) -> "Session":
    """
    Public entry point. Creates and returns a new Session.

        session = holmes.investigate(dataset, model=my_model)
    """

def compare(session_a: "Session", session_b: "Session") -> "SessionDiff":
    """Public entry point for cross-session diffing."""

def load_session(path_or_id: str) -> "Session":
    """Public entry point for restoring a saved session."""
```

---

## How This Maps to the Earlier Reasoning Graph Design

| Session surface              | Reasoning Graph module it delegates to              |
|-------------------------------|-------------------------------------------------------|
| `session.investigate()`       | `construction/builder.py`, all subsystem collectors  |
| `session.collect()`           | `evidence/matcher.py` (if `link_to_graph=True`)       |
| `session.graph.hypotheses()`  | `graph/graph.py` query methods                       |
| `session.graph.trace()`       | `query/tracer.py`                                     |
| `session.graph.simulate()`    | `graph/nodes.py: InterventionNode` + ISS              |
| `session.graph.reject/boost()`| `hypothesis/lifecycle.py` + belief calibration store  |
| `session.findings`            | `query/ranker.py`, `query/exporter.py`                |
| `session.ask()`                | `construction/abductive.py`'s LLM client, reused      |
| `session.tell()`               | `graph/nodes.py: ContextNode` creation                |

The session is intentionally a thin, ergonomic shell — all the real
reasoning logic stays in the `reasoning_graph` package. This keeps
the SDK's user-facing surface stable even as the underlying inference
machinery evolves.
