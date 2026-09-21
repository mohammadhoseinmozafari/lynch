from __future__ import annotations
from typing import Optional
import pandas as pd

from core.visualizer.missingness import RowsMissingnessVisualizer
from core.analyzer import Analyzer ,  AnalysisResult, AnalyzerType


class RowsMissingness(Analyzer):
    """Compute row-level missingness metrics."""

    id = "dataset.misssingness.rates.rows"
    capability = "missingness.row_rates"
    requires = {"profile.base"}
    provides = {capability}
    analyzer_type = AnalyzerType.DATASET

    def __init__(self) -> None:
        super().__init__()



        
        self.visualizer = RowsMissingnessVisualizer()


    
    def mask(self , df:pd.DataFrame) -> pd.DataFrame:
        return df.isna()
    
    def counts(self, df: pd.DataFrame):

        return self.mask(df).sum(axis=1).astype(int)


    def rates(self, df:pd.DataFrame) -> pd.Series[float]:
        n_cols = self.n_cols(df)
        return (
            self.counts(df) / n_cols
            if n_cols > 0
            else pd.Series(0.0, index=df.index, dtype=float)
        )

    def n_cols(self, df:pd.DataFrame) -> int:
        return  df.shape[1]

    def full_missing(self, df : pd.DataFrame, sample_size: Optional[int] = None) -> AnalysisResult:

        mask = self.rates (df) == 1.0
        indices = self.rates(df).index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        raw = {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
        }

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_full_missing,
        )

    def missing_above(
        self,
        df : pd.DataFrame,
        threshold: float,
        sample_size: Optional[int] = None,
    ) -> AnalysisResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        rates = self.rates(df)
        mask = rates >= threshold
        indices = rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        raw = {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
            "threshold": threshold,
            "direction": "above",
            "total_rows": int(len(rates)),
        }

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_threshold,
        )

    def missing_below(
        self,
        df : pd.DataFrame,
        threshold: float,
        sample_size: Optional[int] = None,
    ) -> AnalysisResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        rates = self.rates(df)
        mask = rates <= threshold
        indices = rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        raw = {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
            "threshold": threshold,
            "direction": "below",
            "total_rows": int(len(rates)),
        }

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_threshold,
        )

    def completeness(self, df: pd.DataFrame) -> AnalysisResult:
        hist = self.counts(df).value_counts().sort_index()
        raw = pd.DataFrame({
            "n_missing_fields": hist.index,
            "n_rows": hist.values,
        })

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_completeness,
        )

    def summary(self, df: pd.DataFrame, thresholds=None) -> AnalysisResult:
        rates = self.rates(df)
        raw = {
            "total_rows": int(len(df)),
            "full_missing_rows": int((rates == 1.0).sum()),
        }

        if thresholds is not None:
            above = thresholds.get("above")
            if above is not None:
                if not 0.0 <= above <= 1.0:
                    raise ValueError(
                        "thresholds['above'] must be between 0.0 and 1.0"
                    )
                raw["missing_rows_above"] = int((rates >= above).sum())

            below = thresholds.get("below")
            if below is not None:
                if not 0.0 <= below <= 1.0:
                    raise ValueError(
                        "thresholds['below'] must be between 0.0 and 1.0"
                    )
                raw["missing_rows_below"] = int((rates <= below).sum())

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_summary,
        )

    def stats(self, df: pd.DataFrame) -> AnalysisResult:
        values = self.rates(df).dropna().astype(float)

        metrics = ["mean", "median", "std", "min", "max", "p90", "p95", "p99"]

        if values.empty:
            raw = pd.Series(0.0, index=metrics, name="rates")
        else:
            raw = pd.Series(
                {
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "p90": values.quantile(0.90),
                    "p95": values.quantile(0.95),
                    "p99": values.quantile(0.99),
                },
                name="rates",
            ).astype(float)

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_stats,
        )