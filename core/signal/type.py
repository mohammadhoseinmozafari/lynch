from enum import Enum

class SignalType(str, Enum):
    CRITICAL_MISSING_RATE   = "critical_missing_rate"
    HIGH_MISSING_RATE       = "high_missing_rate"
    MODERATE_MISSING_RATE   = "moderate_missing_rate"
    LOW_MISSING_RATE        = "low_missing_rate"
    COLUMN_INFORMATION_RETAINED         = "column_information_retained"  # This captures "how much usable data remains".
    MISSINGNESS_ESTIMATION_UNCERTAINTY  = "missingness_estimation_uncertainty" # high variance → not enough data to trust missing rate

      

class HealthSignalType (str, Enum) :
    # FEATURE MISSINGNESS HEALTH
    FEATURE_COMPLETENESS            = "feature_completeness"            #   ~Formula : 1- missing_rate
    FEATURE_MISSINGNESS             = "feature_missingness"             #   ~Formula : missing_rate
    FEATURE_USABILITY               = "feature_usability"               #   ~Formula : 1- (missing_rate / threshold)
    FEATURE_OBSERVATION_CONFIDENCE  = "feature_observation_confidence"  #   ~Formula : 1- (missing_rate / threshold)
    
    # ROWS MISSINGNESS HEALTH
    DATASET_ROW_INTEGRITY       =   "dataset_row_integrity"         #   *This shows the integrity of the rows. 
                                                                    #   ~Formula: integrity = 1 - (a x rows_full_missing_rate) - (b x rows_highly_missing_rate)
    COMPLETE_ROW_FAILURE_RATE   =   "complete_row_failuer_rate"     #   *This shows rate of the rows which are completely missing 
    PARTIAL_ROW_FAILURE_RATE    =   "partial_row_failure_rate"      #   *This shows rate of the rows which are highly missing 
    
    # *DATASET MISSINGNESS HEALTH*  Input:  ColsMissingnessDistribution 
    DATASET_COMPLETENESS    =   "dataset_completeness"              #   *This shows the average completeness in dataset
                                                                    #   ~Formula: integrity = 1 - mean_missing_rate
    DATASET_MISSINGNESS     =   "dataset_missingness"               #   *This shows the mean of missing rate in dataset    : mean_missing_rate

    WORST_FEATURE_HEALTH    =   "worth_feature_health"              #   *This shows the health of the feature with highest missing rate
                                                                    #   ~Formula: integrity = 1 - max_missing_rate
    BEST_FEATURE_HEALTH     =   "best_feature_health"               #   *This shows the health of the feature with lowest missing rate  
                                                                    #   ~Formula: integrity = 1 - max_missing_rate

    # *Row distribution health* Input: RowsMissingnessDistribution 
    TYPICAL_ROW_HEALTH      =   "typical_row_health"                #   *These answer: “What is the typical row quality?” 
                                                                    #   ~Formula : row_completeness= 1−mean_missing_rate
    WORST_CASE_ROW_HEALTH   =   "worst_case_row_health"             #   *if max_missing_rate ≈ 1 → some rows are completely unusable 
                                                                    #   ~Formula: 1−max_missing_rate
    BEST_CASE_ROW_HEALTH    =   "best_case_row_health"              #   *This helps detect: partial ingestion failures, heterogeneous pipelines 
                                                                    #   ~Formula: 1−min_missing_rate
    TAIL_HEALTH  = "tail_health"                                    #   *Useful for monitoring.
                                                                    #   ~Formula: 1 - p95_missing_rate

class StructuralSignalType (str, Enum) :
    
    #COLUMN MISSINGNESS SIGNALS
    LOCALIZED_FEATURE_FAILURE   =    "localized_feature_failure"    #   *One or a few columns are much worse than the rest.  
                                                                    #   ~Formula : z_score = (column_missing_rate - dataset_mean) / dataset_std z_score > 2
    GLOBAL_FEATURE_DEGRADATION  =    "global_feature_degradation"   #   *Most columns have similar missing rates. The entire feature space is degraded.     
                                                                    #   ~Formula : dataset_std < threshold and mean_missing_rate > threshold
    DOMINANT_FEATURE_FAILURE    =    "dominant_feature_failure"     #   *One feature contributes a very large fraction of all missing values.
                                                                    #   ~Formula : missing_count(feature) / total_missing_cells
    FEATURE_MISSINGNESS_OUTLIER =    "feature_missingness_outlier"  #   *The feature is an outlier in missingness.
                                                                    #   ~Formula : IQR Rule
    # ROWS MISSINGNESS SIGNALS
    COMPLETE_ROW_CORRUPTION     =    "complete_row_corruption"      #   *Entire records are empty. 
                                                                    #   ~Formula : full_missing_rows_rate > threshold

    PARTIAL_ROW_CORRUPTION      =    "partial_row_corruption"       #   *Many rows have high but incomplete missingness. high_missing_rate_rows_rate > threshold
                                                                    #   ~Formula :  high_missing_rate_rows_rate > threshold    
    MIXED_ROW_CORRUPTION        =    "mixed_row_corruption"         #   *Both complete and partial failures exist.
                                                                    #   ~Formula :  full_rate > t1 and high_rate > t2        
    # ROWS DIST SIGNALS
    UNIFORM_ROW_QUALITY         =   "uniform_row_quality"           #   *Almost every row has similar missingness.
                                                                    #   ~Formula :  std < threshold        
    HETROGENEOUS_ROW_QUALITY    =   "hetrogeneous_row_quality"      #   *Large variability. 
                                                                    #   ~Formula :  std > threshold        
    HEAVY_TAIL_ROW_CORRUPTION   =   "heavy_tail_row_corruption"     #   *A small fraction of rows are extremely damaged.
                                                                    #   ~Formula :  p99 - median > threshold     
    SKEWED_ROW_CORRUPTION       =   "skewed_row_corruption"         #   *Large difference indicates asymmetry.
                                                                    #   ~Formula :  mean - median
    EXTREME_ROW_FAILURE         =   "extreme_row_failure"           #   ~Formula :  max > threshold

    # COLs DIST SIGNALS\
    UNIFORM_FEATURE_QUALITY         =   "uniform_feature_quality"           #   *Almost every column has similar missingness.
                                                                            #   ~Formula :  std < threshold        
    HETROGENEOUS_FEATURE_QUALITY    =   "hetrogeneous_feature_quality"      #   *Large variability among columns missingness.
                                                                            #   ~Formula :  std < threshold        
    HEAVY_TAIL_FEATURE_CORRUPTION   =   "heavy_tail_feature_corruption"     #   *A small fraction of columns are extremely damaged.
                                                                            #   ~Formula :  p99 - median > threshold     
    SKEWED_FEATURE_CORRUPTION       =   "skewed_feature_corruption"         #   *Large difference indicates asymmetry.
                                                                            #   ~Formula :  mean - median
    EXTREME_FEATURE_FAILURE         =   "extreme_feature_failure"           #   ~Formula :  max > threshold
    



class BehavioralSignalType(str, Enum):

    # temporal behavior
    MISSINGNESS_DRIFT       = "missingness_drift"                           #   *Detects increasing or decreasing missingness.
    MISSINGNESS_SPIKE       = "missingness_spike"                           #   *Detects sudden change in missingness.
    MISSINGNESS_REGRESSION  = "missingness_regression"                      #   *Shows improvment signal in missingness.  
    MISSINGNESS_SEASONALITY = "missingness_seasonality"                     

    # system-induced behavior
    PIPELINE_DEPENDENCY_FAILURE = "pipeline_dependency_failure"             #   *Detects upstream system failure causing missingness.
    DOWNSTREAM_PROPAGATION      = "downstream_propagation"                  #   *Detects downstream system failure causing missingness.

    # segment/user behavior
    SEGMENT_BIASED_MISSINGNESS  = "segment_biased_missingness"              #   *Detects if missingness is concentrated in a subgroup.

    # stability behavior
    MISSINGNESS_VOLATILITY      = "missingness_volatility"                  #   *Signal that measures the instability over time. (using std)
    MISSINGNESS_STABILITY_BREAK = "missingness_stability_break"             #   *Signal that stability has broken
