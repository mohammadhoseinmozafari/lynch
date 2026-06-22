from enum import Enum

class SignalType(str, Enum):
    CRITICAL_MISSING_RATE = "critical_missing_rate"
    HIGH_MISSING_RATE =  "high_missing_rate"
    MODERATE_MISSING_RATE = "moderate_missing_rate"
    LOW_MISSING_RATE = "low_missing_rate"
    COLUMN_INFORMATION_RETAINED = "column_information_retained"  # This captures "how much usable data remains".
    MISSINGNESS_ESTIMATION_UNCERTAINTY = "missingness_estimation_uncertainty" # high variance → not enough data to trust missing rate

      

class HealthSignalType (str, Enum) :

    # FEATURE MISSINGNESS HEALTH
    FEATURE_COMPLETENESS = "feature_completeness"
    FEATURE_MISSINGNESS  = "feature_missingness"
    FEATURE_USABILITY    = "feature_usability"
    FEATURE_OBSERVATION_CONFIDENCE = "feature_observation_confidence"
    
    # ROWS MISSINGNESS HEALTH
    DATASET_ROW_INTEGRITY  =    "dataset_row_integrity"
    COMPLETE_ROW_FAILURE_RATE = "complete_row_failuer_rate"
    PARTIAL_ROW_FAILURE_RATE =  "partial_row_failure_rate"
    
    # DATASET MISSINGNESS HEALTH
    DATASET_COMPLETENESS =  "dataset_completeness"
    DATASET_MISSINGNESS =   "dataset_missingness"
    WORST_FEATURE_HEALTH  = "worth_feature_health"
    BEST_FEATURE_HEALTH =   "best_feature_health"
    TYPICAL_ROW_HEALTH =    "typical_row_health"
    WORST_CASE_ROW_HEALTH = "worst_case_row_health"
    TAIL_HEALTH  = "tail_health"


class StructuralSignalType (str, Enum) :
    #COLUMN MISSINGNESS SIGNALS
    LOCALIZED_FEATURE_FAILURE   =    "localized_feature_failure"     # One or a few columns are much worse than the rest.
    GLOBAL_FEATURE_DEGRADATION  =    "global_feature_degradation"    # Most columns have similar missing rates.
    DOMINANT_FEATURE_FAILURE    =    "dominant_feature_failure"
    FEATURE_MISSINGNESS_OUTLIER =    "feature_missingness_outlier"   # The feature is an outlier in missingness.
    
    # ROWS MISSINGNESS SIGNALS
    COMPLETE_ROW_CORRUPTION     =    "complete_row_corruption"       # Entire records are empty. full_missing_rows_rate > threshold
    PARTIAL_ROW_CORRUPTION      =    "partial_row_corruption"        # Many rows have high but incomplete missingness. high_missing_rate_rows_rate > threshold
    MIXED_ROW_CORRUPTION        =    "mixed_row_corruption"
    
    # ROWS DIST SIGNALS
    UNIFORM_ROW_QUALITY         =   "uniform_row_quality"           #Almost every row has similar missingness. std < threshold
    HETROGENEOUS_ROW_QUALITY    =   "hetrogeneous_row_quality"      # Large variability.    std > threshold
    HEAVY_TAIL_ROW_CORRUPTION   =   "heavy_tail_row_corruption"     # A small fraction of rows are extremely damaged.
    SKEWED_ROW_CORRUPTION       =   "skewed_row_corruption"
    EXTREME_ROW_FAILURE         =   "extreme_row_failure"

    # COLs DIST SIGNALS\
    UNIFORM_FEATURE_QUALITY         =   "uniform_feature_quality"
    HETROGENEOUS_FEATURE_QUALITY    =   "hetrogeneous_feature_quality"
    HEAVY_TAIL_FEATURE_CORRUPTION   =   "heavy_tail_feature_corruption"    
    SKEWED_FEATURE_CORRUPTION       =   "skewed_feature_corruption"
    EXTREME_FEATURE_FAILURE         =   "extreme_feature_failure"
    



class BehavioralSignalType(str, Enum):

    # temporal behavior
    MISSINGNESS_DRIFT = "missingness_drift"
    MISSINGNESS_SPIKE = "missingness_spike"
    MISSINGNESS_REGRESSION = "missingness_regression"
    MISSINGNESS_SEASONALITY = "missingness_seasonality"

    # system-induced behavior
    PIPELINE_DEPENDENCY_FAILURE = "pipeline_dependency_failure"
    DOWNSTREAM_PROPAGATION = "downstream_propagation"

    # segment/user behavior
    SEGMENT_BIASED_MISSINGNESS = "segment_biased_missingness"

    # stability behavior
    MISSINGNESS_VOLATILITY = "missingness_volatility"
    MISSINGNESS_STABILITY_BREAK = "missingness_stability_break"
