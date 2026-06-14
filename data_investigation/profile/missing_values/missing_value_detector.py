from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from core.domain.entities.evidence import Evidence
from data_investigation.base.detector import BaseAnalyzer




class MissingValueAnalysisConfig(BaseModel) :
    # Thresholds
    critical_missing_rate_threshold : float = Field(0.8, ge=0.0, le=1.0)
    high_missing_rate_threshold : float = Field(0.6, ge=0.0, le=1.0)
    moderate_missing_rate_threshold : float = Field(0.4, ge=0.0, le=1.0)
    low_missing_rate_threshold : float = Field(0.2, ge=0.0, le=1.0)
    very_low_missing_rate_threshold : float = Field(0.0, ge=0.0, le=1.0)

    row_missingness_sample_size: int = Field (100, ge=0.0)

    @model_validator(mode = 'after')
    def validate_thresholds (self) -> 'MissingValueAnalysisConfig':
        """Validate thresholds are in descending order"""
        thresholds = [
            ('critical', self.critical_missing_rate_threshold),
            ('high', self.high_missing_rate_threshold),
            ('moderate', self.moderate_missing_rate_threshold),
            ('low', self.low_missing_rate_threshold),
            ('very_low', self.very_low_missing_rate_threshold)
        ]

        for i in range(len(thresholds) - 1):
            curr_name, curr_value = thresholds[i]
            next_name, next_value = thresholds[i + 1]
            
            if curr_value <= next_value:
                raise ValueError(
                    f'Thresholds must be strictly decreasing: '
                    f'{curr_name} ({curr_value}) <= {next_name} ({next_value})'
                )
        
        return self

    def get_thresholds (self) -> Dict[str, float]:
        return {
            "critical": self.critical_missing_rate_threshold,
            "high":     self.high_missing_rate_threshold,
            "moderate": self.moderate_missing_rate_threshold,
            "low" :     self.low_missing_rate_threshold,
            "very_low" : self.very_low_missing_rate_threshold
        }
    
@dataclass
class AnalysisContext :
    config: MissingValueAnalysisConfig
    dataset : pd.DataFrame

class MissingRateAnalyzer (BaseAnalyzer):

    def __init__(self, evidence_builder) -> None:
        super().__init__()

        self._evidence_builder = evidence_builder
    
    
    
    def analyze(self, context: AnalysisContext) -> List[Evidence]:

        evidences : List[Evidence] = []

        column_missingness_evidences = self.analyze_columns(context)
        row_missingness_evidences = self.analyze_rows(context)

        evidences.extend(column_missingness_evidences)
        evidences.extend(row_missingness_evidences)

        return evidences




    
    def analyze_columns (self, context: AnalysisContext)  -> List[Evidence]:
        
        data = context.dataset
        config = context.config
        thresholds = config.get_thresholds()
        
           
        evidences : List[Evidence]  = []

        columns_missing_rates = self.compute_columns_missing_rates(data)
        columns_missingness = self.classify_columns_missingness(columns_missing_rates, thresholds)
        evidences = self._evidence_builder.build_column_missingness_evidences(columns_missingness)
            
        return evidences
    
    
    def analyze_rows (self, context : AnalysisContext) -> List[Evidence]:

        data = context.dataset
        thresholds = context.config.get_thresholds()
        sample_size = context.config.row_missingness_sample_size

        evidences : List[Evidence]  = []

        rows_missing_rates = self.compute_rows_missingness(data)
        rows_missingness= self.classify_rows_missingness(rows_missing_rates, thresholds, len(data), sample_size)
        
        evidences = self._evidence_builder.build_row_missingness_evidences(rows_missingness)
        
        return evidences
    

    
    def classify_rows_missingness (self, rows_missing_rates: pd.Series, thresholds: Dict[str, float], dataset_size : int , sample_size : int) -> Optional[Dict[str, Any]]:
        
        threshold = thresholds.get("high", 0.6)
        
        high_missing_rate_rows = self.classify_high_missingness_rows(rows_missing_rates, threshold)
        
        if high_missing_rate_rows is not None:
            affected_rows_sample = high_missing_rate_rows.sample(sample_size)     
            total_affected = len(high_missing_rate_rows)
            affected_percentage = total_affected / dataset_size * 100

            result : Dict = {
                "affected_rows_sample": affected_rows_sample,
                "total_affected" : total_affected,
                "affected_percentage": affected_percentage
                
            }

            return result
        return None
        
    
    def classify_columns_missingness (self, columns_missing_rates: pd.Series, thresholds: Dict[str, float]) -> Optional[Dict[str, float]]:

        critical_missing_rate_columns = self.classify_critical_missingness_columns(columns_missing_rates, thresholds)
        high_missing_rate_columns = self.classify_high_missingness_columns(columns_missing_rates, thresholds)
        moderate_missing_rate_columns = self.classify_moderate_missingness_columns(columns_missing_rates, thresholds)        
        low_missing_rate_columns  = self.classify_low_missingness_columns(columns_missing_rates, thresholds)
        very_low_missing_rate_columns  = self.classify_very_low_missingness_columns(columns_missing_rates, thresholds)


        result : Dict = {}

        if critical_missing_rate_columns is not None:
            result["critical_missing_rate"] = critical_missing_rate_columns.to_dict()
    
        if high_missing_rate_columns is not None:
            result["high_missing_rate"] = high_missing_rate_columns.to_dict()
        
        if moderate_missing_rate_columns is not None:
            result["moderate_missing_rate"] = moderate_missing_rate_columns.to_dict()
        
        if low_missing_rate_columns is not None:
            result["low_missing_rate"] = low_missing_rate_columns.to_dict()

        if very_low_missing_rate_columns is not None:
            result["very_low_missing_rate"] = very_low_missing_rate_columns.to_dict()

            
        
        return result if result else None
    

    def classify_critical_missingness_columns(self, columns_missing_rates : pd.Series, thresholds: Dict[str, float]) -> Optional[pd.Series]:
        
        critical_threshold: float = thresholds.get("critical", 0.8)
        critical_missing_rate_columns = columns_missing_rates[columns_missing_rates > critical_threshold]
        
        return critical_missing_rate_columns if (not critical_missing_rate_columns.empty) else None




    def classify_high_missingness_columns(self, columns_missing_rates : pd.Series, thresholds: Dict[str, float])-> Optional[pd.Series]:
        
        critical_threshold: float = thresholds.get("critical", 0.8)        
        high_threshold : float = thresholds.get("high", 0.6)

        high_missing_rate_columns = columns_missing_rates[
            (columns_missing_rates > high_threshold)&  
            (columns_missing_rates <= critical_threshold)
            
        ]

        return high_missing_rate_columns if (not high_missing_rate_columns.empty) else None



    def classify_moderate_missingness_columns(self, columns_missing_rates : pd.Series, thresholds: Dict[str, float])-> Optional[pd.Series]:
        
        high_threshold : float = thresholds.get("high", 0.6)
        moderate_threshold : float  = thresholds.get("moderate", 0.4)
        
        moderate_missing_rate_columns = columns_missing_rates[
            (columns_missing_rates > moderate_threshold)&  
            (columns_missing_rates <= high_threshold)
            
        ]

        return moderate_missing_rate_columns if (not moderate_missing_rate_columns.empty) else None
    
    def classify_low_missingness_columns(self, columns_missing_rates : pd.Series, thresholds: Dict[str, float])-> Optional[pd.Series]:
        
        moderate_threshold : float  = thresholds.get("moderate", 0.4)
        low_threshold : float = thresholds.get("low", 0.2)
        
        low_missing_rate_columns  = columns_missing_rates[
            (columns_missing_rates > low_threshold)&  
            (columns_missing_rates <= moderate_threshold)
            
        ]

        return low_missing_rate_columns if (not low_missing_rate_columns.empty) else None

    def classify_very_low_missingness_columns(self, columns_missing_rates : pd.Series, thresholds: Dict[str, float])-> Optional[pd.Series]:
        
        low_threshold : float = thresholds.get("low", 0.2)
        very_low_threshold: float = thresholds.get("very_low", 0)       
        
        very_low_missing_rate_columns  = columns_missing_rates[
            (columns_missing_rates > very_low_threshold)&  
            (columns_missing_rates <= low_threshold)
            
        ]
        return very_low_missing_rate_columns if (not very_low_missing_rate_columns.empty) else None
    
    def classify_high_missingness_rows (self, rows_missing_rates: pd.Series, threshold: float) -> Optional[pd.Series]:
        high_missing_rate_rows_mask =  rows_missing_rates >= threshold
        high_missing_rate_rows = rows_missing_rates[high_missing_rate_rows_mask]
        return high_missing_rate_rows

    



    
   
    
    

    
    
    

        


    def compute_columns_missing_rates (self, df: pd.DataFrame) -> pd.Series:

        return df.isnull().mean()

    
    def compute_rows_missingness (self, df: pd.DataFrame) -> pd.Series:
       
       return df.isnull().mean(axis=1)
    
    def analyze_column(self, column  , threshold) :
        pass
    

        
    
    

