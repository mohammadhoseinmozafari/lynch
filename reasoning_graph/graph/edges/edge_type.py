from enum import Enum


class EdgeType(str, Enum):
    EVIDENTIAL    = "evidential"    # ObservationNode → HypothesisNode
    CAUSAL        = "causal"        # HypothesisNode → HypothesisNode (cause → effect)
    COMPETITIVE   = "competitive"   # HypothesisNode ↔ HypothesisNode (mutual exclusion)
    CORROBORATING = "corroborating" # HypothesisNode → HypothesisNode (non-causal support)
    CONTEXTUAL    = "contextual"    # ContextNode → HypothesisNode (prior shift)
    PREREQUISITE  = "prerequisite"  # HypothesisNode → HypothesisNode (must resolve first)
    SPAWNED       = "spawned"       # HypothesisNode → HypothesisNode (parent → child cascade)