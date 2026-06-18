from enum import Enum


class NodeType(str , Enum):
    EVIDENCE  = "evidence"   # raw evidence / measurement
    HYPOTHESIS   = "hypothesis"    # candidate explanation
    CONTEXT      = "context"       # background knowledge / metadata
    INTERVENTION = "intervention"  # counterfactual / what-if action
    FINDING      = "finding"       # graduated, confirmed conclusion

