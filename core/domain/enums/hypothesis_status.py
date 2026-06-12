from enum import Enum

class HypothesisStatus (str, Enum):
    ACTIVE = "ACITVE"
    REJECTED = "REJECTED"
    CONFIRMED = "CONFIRMED"
