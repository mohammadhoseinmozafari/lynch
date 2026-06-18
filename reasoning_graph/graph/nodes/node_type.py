from enum import Enum


class NodeType(str , Enum):
    EVIDENCE  = "evidence"          # raw evidence / measurement
    PATTERN = "pattern"             # pattern that was recognized from set of evidences
    HYPOTHESIS   = "hypothesis"     # candidate explanation
    CONTEXT      = "context"        # background knowledge / metadata
    INTERVENTION = "intervention"   # counterfactual / what-if action
    FINDING      = "finding"        # graduated, confirmed conclusion

