from enum import Enum


class NodeStatus(str, Enum):
    INACTIVE      = "inactive"       # exists but not yet activated
    ACTIVE       = "active"        # currently being evaluated
    CONFIRMED    = "confirmed"     # confidence above threshold
    REJECTED     = "rejected"      # confidence below threshold
    INCONCLUSIVE = "inconclusive"  # evidence conflict, needs more info
    MERGED       = "merged"        # unified with another node
    GRADUATED    = "graduated"     # promoted to a Finding


