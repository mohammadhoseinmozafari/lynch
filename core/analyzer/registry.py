from __future__ import annotations
from typing import Dict

from core.analyzer.missingness import ColumnsMissingness, RowsMissingness



from .analyzer import Analyzer


class AnalyzerRegistry :
    def __init__(self) -> None:
        self.items : Dict [str, Analyzer] = {}

    
    def register (self , analyzer : Analyzer)  -> None:
        self.check_duplicate(analyzer)
        self.items[analyzer.id] = analyzer

    def get (self, id : str) -> Analyzer:
        try :
            return self.items[id]
        except KeyError:
            raise ValueError(
                f"Unkown analyzer : {id}"
            )
    def check_duplicate (self, analyzer: Analyzer) -> None:
        if analyzer.id in self.items:
            raise ValueError (
                f"Analyzer already registered:{analyzer.id}"
            )
        
    def initialize_runtime (self) -> None:

        self.register(
            ColumnsMissingness()
        )
        self.register(
            RowsMissingness()
        )