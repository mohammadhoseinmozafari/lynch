# holmes/application/service.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.analyzer import AnalyzerRegistry
from core.analyzer.missingness import  RowsMissingness, ColumnsMissingness

from core.resource import Resource, ResourceRegistry



@dataclass(frozen= True)
class AnalysisRequest:
    analyzer_id : str
    inputs : Dict[str, str]
    parameters: dict[str, Any] = field(
        default_factory=dict
    )



class HolmesService:
    def __init__(
        self,
        *,
        resources : ResourceRegistry,
        analyzers: AnalyzerRegistry,
    ) -> None:
        self.resources = resources
        self.analyzers = analyzers

    # def analyze_missing_rates(self , dataset_id : str, scope):
    #     dataset = self.resources.load(dataset_id)
    #     final  = {}
    #     analyzers  = {
        
    #         "column": ColumnMissingRateAnalyzer(),
    #         "row": RowsMissingRateAnalyzer(),
    #         "column_distribution": ColumnDistributionMissingRateAnalyzer(),
    #         "row_distribution": RowsDistributionMissingRateAnalyzer()
    #     }
    #     if scope is None:
    #         for analyzer_name , analyzer in analyzers.items():
    #             analysis = analyzer.analyze(dataset)
    #             final[analyzer_name] = analysis  
                
    #     else:
    #         for s in scope:
    #             if s

    #     return analysis