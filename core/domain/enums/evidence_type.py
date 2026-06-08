from enum import Enum


class EvidenceType(str, Enum):
    STATISTICAL_TEST = "STATISTICAL_TEST"
    DISTRIBUTION_COMPARISON = "DISTRIBUTION_COMPARISON"
      