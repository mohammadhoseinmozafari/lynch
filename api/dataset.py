# holmes/api/dataset.py

from __future__ import annotations

from typing import Any, Collection, Optional

from core.analyzer.missingness import MissingnessScope
from core.service import AnalysisRequest, HolmesService


class RequestResolver:

    _MISSINGNESS_SCOPES = {
        MissingnessScope.COLUMN: {
            "dataset.misssingness.rates.columns",
        },

        MissingnessScope.ROW: {
            "dataset.misssingness.rates.rows",
        },


    }

    def resolve_missingness_scopes(self, scope : Optional[Collection[MissingnessScope] | MissingnessScope] ):
        if scope is not None:
            scopes = {MissingnessScope(s) for s in scope}
            return scopes
        return self._MISSINGNESS_SCOPES        


class Dataset:
    def __init__(
        self,
        *,
        id: str,
        name: str,
        service: HolmesService,
    ) -> None:
        self._id = id
        self._name = name
        self._service = service
        self._request_resolver = RequestResolver()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def missing_rates(self, scope: Optional[Collection[MissingnessScope] | MissingnessScope] = None ) -> Any:
        scopes = self._request_resolver.resolve_missingness_scopes(scope)  
        return self._service.analyze_missing_rates(self._id , scopes)
    
    def missing_clusters(self) -> Any:
        return self._service.run(
            AnalysisRequest(
                analyzer_id="dataset.missing_clusters",
                inputs={
                    "dataset": self.id
                }
            )
        )

    def missing_value_correlation(self) -> Any:
        return self._service.run(
            AnalysisRequest(
                analyzer_id="dataset.missing_value_correlation",
                inputs={
                    "dataset": self.id
                }
            )
        )

