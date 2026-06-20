from enum import Enum

class SignalType(str, Enum):
    CRITICAL_MISSING_RATE = "critical_missing_rate"
    HIGH_MISSING_RATE =  "high_missing_rate"
    MODERATE_MISSING_RATE = "moderate_missing_rate"
    LOW_MISSING_RATE = "low_missing_rate"
    COLUMN_INFORMATION_RETAINED = "column_information_retained"  # This captures "how much usable data remains".
    MISSINGNESS_ESTIMATION_UNCERTAINTY = "missingness_estimation_uncertainty" # high variance → not enough data to trust missing rate

      