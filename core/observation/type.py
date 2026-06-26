from enum import Enum


class ObservationType (str, Enum):
    HIGH_COLUMN_MISSINGNESS             = "high_coulmn_missingness"
    MODERATE_COLUMN_MISSINGNESS         = "moderate_column_missingness"
    LOW_COLUMN_MISSINGNESS              = "low_column_missingness"
    COLUMN_MISSINGNESS                  = "column_missingness" 
    ROWS_MISSINGNESS                    = "rows_missingness" 
    ROWS_MISSINGNESS_DISTRIBUTION       = "rows_missingness_distribution"
    COLUMNS_MISSINGNESS_DISTRIBUTION    = "columns_missingness_distribution"    
