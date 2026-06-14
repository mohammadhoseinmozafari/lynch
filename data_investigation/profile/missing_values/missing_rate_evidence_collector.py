from __future__ import annotations

from typing import List
import pandas as pd
from core.domain.entities.evidence import Evidence
from core.domain.enums.evidence_type import EvidenceType
from data_investigation.base.evidence_collector import BaseEvidenceCollector
from data_investigation.profile.missing_values.profiler import ColumnMissingRateProfile, MissingRateProfiler, MissingnessDistributionProfile, MissingnessDistributionProfile, RowsMissingRateProfile, RowsMissingRateProfile







class MissingRateEvidenceCollector(BaseEvidenceCollector):
    
    def __init__(self, profiler: MissingRateProfiler) -> None:
        super().__init__()
        
        self._profiler = profiler

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
        
        columns_missingness_profile= self._profiler.profile_columns(df)

        for column, profile in columns_missingness_profile.items():
            column_missingness_evidence = self.build_columns_missingness_evidence(profile)

            evidences.append(column_missingness_evidence)

        return evidences
    
    
    def collect_rows_evidences (self, df: pd.DataFrame, sample_size : int) -> List[Evidence]:
        
        evidences : List[Evidence] = []
        rows_missingness_profile = self._profiler.profile_rows(df, sample_size)

        rows_missingness_evidence = self.build_rows_missingness_evidence(rows_missingness_profile)

        evidences.append(rows_missingness_evidence)
        
        return evidences
    
    def collect_distribution_evidences (self, df: pd.DataFrame) -> List[Evidence]:
        
        evidences : List[Evidence] = []
        missingness_distribution_profile = self._profiler.profile_distribution(df)

        distribution_evidence = self.build_distribution_evidence (missingness_distribution_profile)

        evidences.append(distribution_evidence)
        
        return evidences
    
   
    
    def build_columns_missingness_evidence (self, column_missingness_profile :  ColumnMissingRateProfile) -> Evidence:

        return Evidence(
            type = EvidenceType.COLUMN_MISSINGNESS,
            payload = column_missingness_profile
        )

        
            

    def build_rows_missingness_evidence (self, rows_missingness_profile : RowsMissingRateProfile) -> Evidence:
        
        return Evidence (
            type = EvidenceType.ROWS_MISSINGNESS,
            payload = rows_missingness_profile
        )

    def build_distribution_evidence (self, missingness_distribution_profile : MissingnessDistributionProfile) -> Evidence:
        
        return Evidence (
            type = EvidenceType.MISSINGNESS_DISTRIBUTION,
            payload = missingness_distribution_profile
        )






