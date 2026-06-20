
from enum import Enum
from typing import Dict
from ulid import ulid
from datetime import datetime

from reasoning_graph.graph.edges.edge_type import EdgeType
from pydantic import BaseModel, Field


class BaseEdge(BaseModel):
    """
    Every edge in the reasoning graph.
    The weight encodes the strength of the relationship —
    its exact meaning depends on EdgeType (see subclasses).
    """
    id:          str      = Field(default_factory=lambda: str(ulid()))

    edge_type:   EdgeType 

    source_id:   str         # source node ID
    target_id:   str         # target node ID

    weight:      float     # relationship strength [0,1]

    created_at:  datetime = Field(default_factory=datetime.now)
    metadata:    Dict   = Field(default_factory=dict)


class EvidencePolarity (str, Enum):
    SUPPORTS = "supports"
    CONTRADITCTS = "contradicts" 

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
    
    p_e_given_h:      float       # P(E | H)
    p_e_given_not_h:  float       # P(E | ¬H)
    
    polarity:         EvidencePolarity        # "supports" | "contradicts"

    @property
    def likelihood_ratio(self) -> float:
        if self.p_e_given_not_h == 0:
            return float('inf')
        return self.p_e_given_h / self.p_e_given_not_h





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
    
    p_target_given_source:      float      # P(effect | cause is true)
    p_target_given_not_source:  float      # P(effect | cause is false)
    



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
    suppression_strength: float       # [0,1]
    
    shared_evidence_ids:  list[str] = Field(default_factory=list)



class CorroboratingEdge(BaseEdge):
    """
    Non-causal: source being true makes target more plausible.
    Weaker than causal — no mechanism, just co-occurrence pattern
    from prior knowledge.
    """
    edge_type:          EdgeType = EdgeType.CORROBORATING
    
    corroboration_strength: float   # how much confidence boost target gets



class ContextualEdge(BaseEdge):
    """
    From ContextNode to HypothesisNode.
    Shifts the prior of the hypothesis based on background knowledge.

    prior_delta: +0.25 means "in this context, this hypothesis
    starts 25 percentage points more likely than the default prior."
    """
    edge_type:    EdgeType = EdgeType.CONTEXTUAL
    prior_delta:  float       # additive shift to hypothesis prior



class PrerequisiteEdge(BaseEdge):
    """
    Source must reach a minimum confidence before target activates.
    Models logical dependencies between hypotheses.

    Example: LEAKAGE_SUSPECTED must be ACTIVE before
    TEMPORAL_LEAKAGE child hypotheses are worth evaluating.
    """
    edge_type:           EdgeType = EdgeType.PREREQUISITE
    activation_threshold: float     # min confidence of source to unlock target



class SpawnedEdge(BaseEdge):
    """
    Created when a confirmed hypothesis spawns child hypotheses
    via cascade. Tracks the parent-child lineage.
    """
    edge_type:    EdgeType = EdgeType.SPAWNED
    spawn_reason: str         # why this child was spawned