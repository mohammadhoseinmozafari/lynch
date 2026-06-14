

from abc import ABC, abstractmethod
from calendar import c
from importlib.metadata import distribution
from random import sample
from typing import Any, Dict, List, Optional
import pandas as pd
from core import evidence
from core.domain.entities.evidence import Evidence


class BaseEvidenceCollector(ABC):

    @abstractmethod
    def collect (self, context) -> List[Evidence]:
        raise NotImplementedError





class MissingRateEvidenceCollector(BaseEvidenceCollector):
    
    def __init__(self, evidence_builder : BaseEvidenceBuilder, stats_profiler: BaseStatsProfiler) -> None:
        super().__init__()
        self._evidence_builder = evidence_builder
        self._stats_profiler = stats_profiler

    def collect (self, context) -> List[Evidence]:
        
        df : pd.DataFrame = context.dataset
        sample_size : int = context.sample_size

        
        evidences : List[Evidence] = []
        
        columns_evidences = self.collect_columns_evidences(df)
        rows_evidences = self.collect_rows_evidences(df,sample_size)
        distribution_evidences = self.collect_distribution_evidences(df)

        evidences.extend(columns_evidences)
        evidences.extend(rows_evidences)
        evidences.extend(distribution_evidences)

        return evidences
    
    def collect_columns_evidences (self, df : pd.DataFrame) -> List[Evidence]:

        evidences : List[Evidence] = []
        
        columns_missing_stats = self.compute_columns_missing_stats(df)

        columns_missingness_evidence = self._evidence_builder.build_columns_missingness_evidence(columns_missing_stats)

        evidences.append(columns_missingness_evidence)

        return evidences
    
    def collect_distribution_evidences (self, df: pd.DataFrame) -> List[Evidence]:
        
        evidences : List[Evidence] = []
        
        missingness_distribution = self.compute_missingness_distribution (df)

        distribution_evidence = self._evidence_builder.build_distribution_evidence (missingness_distribution)

        evidences.append(distribution_evidence)
        
        return evidences
    
    def collect_rows_evidences (self, df: pd.DataFrame, sample_size : int) -> List[Evidence]:
        
        evidences : List[Evidence] = []

        rows_missing_stats = self.compute_rows_missingness_stats(df, sample_size)

        rows_missingness_evidence = self._evidence_builder.build_rows_missingness_evidence(rows_missing_stats)

        evidences.append(rows_missingness_evidence)
        
        return evidences



    
    
    def compute_columns_missing_stats (self, df : pd.DataFrame) -> Dict[str, Dict[str, Any]]:

        total_rows = len(df)
        missing_counts = df.isna().sum()

        stats = {}

        for column in df.columns:

            missing_count = int(missing_counts[column])
            non_missing_count = total_rows- missing_count

            missing_rate = missing_count/ total_rows if total_rows > 0 else 0.0

            stats[column] = {

                "missing_count" : missing_count,
                "non_missing_count": non_missing_count,
                "total_count" : total_rows,
                "missing_rate": missing_rate
            }
        
        return stats
    
    
    def compute_rows_missingness_stats (self, df: pd.DataFrame, sample_size: int):

        rows_missing_rates = df.isna().mean()
        total_rows = len(df)

        high_missing_rate_rows_stats = self.identify_high_missing_rate_rows (rows_missing_rates, sample_size)
        fully_missing_rows_stats = self.compute_fully_missing_rows_stats (rows_missing_rates, sample_size)   
        

        stats = {
            "high_missing_rate_rows_stats" : high_missing_rate_rows_stats,
            "fully_missing_rows_stats" : fully_missing_rows_stats 
            }

        return stats

    
    def compute_missingness_distribution (self , df : pd.DataFrame) -> Dict[str, float]:
        
        rows_missing_rates = df.isna().mean(axis=1)
        
        return {
        "mean_missing_rate" : float (rows_missing_rates.mean()),
        "median_missing_rate ": float(rows_missing_rates.median()),
        "std_missing_rate" : float(rows_missing_rates.std()),

        "min_missing_rate" : float(rows_missing_rates.min()),
        "max_missing_rate" : float(rows_missing_rates.max()),

        "p90_missing_rate" : float(rows_missing_rates.quantile(0.90)),
        "p95_missing_rate" : float(rows_missing_rates.quantile(0.95)),
        "p99_missing_rate" : float(rows_missing_rates.quantile(0.99)),
        
        }
    
    def compute_fully_missing_rows_stats (self , rows_missing_rates : pd.Series, sample_size : int) -> Optional[Dict[str, Any]]: 
        fully_missing = rows_missing_rates [rows_missing_rates == 1.0]

        if fully_missing.empty:
            return None
        
        sample_indices = fully_missing.sample(sample_size).index.tolist()

        return {
            "count" : len(fully_missing),
            "sample_indices" : sample_indices 
        }
    
    def identify_high_missing_rate_rows (self, rows_missing_rates: pd.Series,  sample_size: int) -> Optional[Dict[str, Any]]:
        
        candidates = rows_missing_rates [rows_missing_rates >= 0.8]
        sample_indices = candidates.sample(sample_size).index.tolist()

        if candidates.empty:
            return None

        return {

            "candidate_count" : len(candidates),
            "sample_indices" : sample_indices
        } 
        




