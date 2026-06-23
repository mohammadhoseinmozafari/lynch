from enum import Enum


class ObservationType (str, Enum):
    COLUMN_MISSINGNESS                  = "column_missingness" 
    ROWS_MISSINGNESS                    = "rows_missingness" 
    ROWS_MISSINGNESS_DISTRIBUTION       = "rows_missingness_distribution"
    COLUMNS_MISSINGNESS_DISTRIBUTION    = "columns_missingness_distribution"    
