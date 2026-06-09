from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from core.case.domain.value_objects.data_profile import DataProfile
from core.domain.entities.evidence import Evidence
from core.domain.entities.finding import Finding
from core.domain.entities.recommendation import EffortLevel, Recommendation
from core.domain.enums.evidence_type import EvidenceType
from core.domain.enums.finding_type import FindingType
from core.domain.enums.severity import Severity
from data_investigation.base.detector import BaseDetector

@dataclass
class MissingValueAnalysisConfig :
    high_missing_rate_threshold : float
    moderate_missing_rate_threshold : float
    
@dataclass
class AnalysisContext :
    config: MissingValueAnalysisConfig
    dataset : pd.DataFrame

class MissingValueDetector (BaseDetector):
    
    def analyze(self, context: AnalysisContext) -> List[Finding]:
        findings : List[Finding] = []

        column_missingness_findings = self.analyze_column_missingness(context)
        row_missingness_findings = self.analyze_row_missingness(context)

        findings.extend(column_missingness_findings)
        findings.extend(row_missingness_findings)

        return findings




    
    def analyze_column_missingness (self, context: AnalysisContext)  -> List[Finding]:
        data = AnalysisContext.dataset
        high_threshold  = context.config.high_missing_rate_threshold
        moderate_threshold = context.config.moderate_missing_rate_threshold

        findings : List[Finding]  = []

        column_missingness = self.compute_column_missingness(data)

        for column , missing_rate in column_missingness.items():

            if missing_rate == 0.0 : 
                continue

            elif missing_rate == 1.0 :

                finding = Finding (
                    type = FindingType.ALL_MISSING_COLUMN,
                    severity= Severity.CRITICAL,
                    confidence= 1.0,
                    title=f"Column '{column}' is completely missing",

                    description= (
                        f"Column '{column} has no valid data (100% missing)"
                    ),
                    evidence= [
                        Evidence(
                            type = EvidenceType.DATA_PROFILE_CHECK,
                            payload={
                                "missing_rate" : 1.0
                            }
                        )
                    ],


                )
            elif missing_rate >= high_threshold:
                finding = Finding(
                    type = FindingType.HIGH_MISSING_VALUE_RATE,
                    severity= Severity.HIGH,
                    confidence= 1.0,
                    title= f"High missing value rate in column '{column}'",
                    description=(
                        f"Column '{column}' has {missing_rate:.1%} missing values. "
                        f"This exceeds the configured high missingness threshold of {high_threshold:.0%}. "
                    ),
                    evidence= [Evidence(
                        type=  EvidenceType.DATA_PROFILE_CHECK,
                        payload= {
                            "missing_rate": missing_rate,
                            "threshold" : high_threshold
                        }
                    )],
                )
                findings.append(finding)

            elif missing_rate >= moderate_threshold :
                finding = Finding(

                    type = FindingType.MODERATE_MISSING_VALUE_RATE,
                    severity= Severity.MEDIUM,
                    confidence= 1.0,
                    title= f"Moderate missing value rate in column '{column}'",
                    description=(
                        f"Column '{column}' has {missing_rate:.1%} missing values. "
                        f"This exceeds the configured moderate missingness threshold of {moderate_threshold:.0%}. "
                    ),
                    evidence= [Evidence(
                        type=  EvidenceType.DATA_PROFILE_CHECK,
                        payload= {
                            "missing_rate": missing_rate,
                            "threshold" : moderate_threshold
                        }
                    )],
                )
                findings.append(finding)
            
        return findings
    
    def analyze_row_missingness (self, context : AnalysisContext) -> List[Finding]:

        data = AnalysisContext.dataset
        threshold = context.config.high_missing_rate_threshold

        findings : List[Finding]  = []

        row_missingness = self.compute_row_missingness(data)
        high_missing_rate_mask = row_missingness >= threshold
        high_missing_rate_idices  = row_missingness[high_missing_rate_mask].index.tolist()
        total_affected = len(high_missing_rate_idices)
        affected_percentage = total_affected / len(data) * 100

        if high_missing_rate_idices :
            finding = Finding(

                    type = FindingType.HIGH_ROW_MISSING_VALUE_RATE,
                    severity= Severity.LOW,
                    confidence= 1.0,
                    title= f"{total_affected} rows have high missing value rate.",
                    description=(
                        f"{total_affected} rows ({affected_percentage:.1f}% of data) have missing percentage ≥ {threshold:.0%}. "
                        f"These rows may need special handling."
                        ),
                    evidence= [Evidence(
                        type=  EvidenceType.DATA_PROFILE_CHECK,
                        payload= {
                            "total_affected_rows": total_affected,
                            "percentage_affected": affected_percentage,
                            "threshold" : threshold,

                        }
                    )],
                )
            findings.append(finding)
        return findings


    
    
    

        


    def compute_column_missingness (self, df: pd.DataFrame) -> pd.Series:

        return df.isnull().mean()

    
    def compute_row_missingness (self, df: pd.DataFrame) -> pd.Series:
       
       return df.isnull().mean(axis=1)
    
    