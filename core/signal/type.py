from enum import Enum

class SignalType(str, Enum):
    HIGH_MISSING_RATE =  "high_missing_rate"
    MODERATE_MISSING_RATE = "moderate_missing_rate"
    LOW_MISSING_RATE = "low_missing_rate"
      