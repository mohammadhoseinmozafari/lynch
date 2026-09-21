

from __future__ import annotations
import pandas as pd

from core.visualizer.missingness import ColumnsMissingnessVisualizer
from core.analyzer import Analyzer ,  AnalysisResult, AnalyzerType


class ColumnsMissingness(Analyzer):
    """Compute per-column missingness metrics."""

    id = "dataset.missingness.rates.columns"

    capability = "missingness.column_rates"

    requires = {"profile.base"}

    provides = {capability}

    analyzer_type = AnalyzerType.DATASET

    def __init__(self) -> None:
        super().__init__()
        self.visualizer = ColumnsMissingnessVisualizer()


    def summary(self, df : pd.DataFrame) -> AnalysisResult:
        
        counts , rates = self._context(df)

        summary = pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": counts.values,
                "missing_rate": rates.values,
                "dtype": [str(t) for t in df.dtypes],
            }
        ).sort_values("missing_rate", ascending=False).reset_index(drop=True)
        
        

        return AnalysisResult(
                raw = summary,
                visualizer = self.visualizer.visualize_summary
            )
            
        

    def stats(self, df: pd.DataFrame) -> AnalysisResult:
        metrics = ["mean", "median", "std", "min", "max", "p90", "p95", "p99"]
        counts , rates = self._context(df)
        if rates.empty:
            stats_df = pd.DataFrame(
                0.0, index=metrics, columns=["rates", "counts"]
            )
        else:
            stats_df = pd.DataFrame(
                {
                    "rates": [
                        rates.mean(),
                        rates.median(),
                        rates.std(),
                        rates.min(),
                        rates.max(),
                        rates.quantile(0.90),
                        rates.quantile(0.95),
                        rates.quantile(0.99),
                    ],
                    "counts": [
                        counts.mean(),
                        counts.median(),
                        counts.std(),
                        counts.min(),
                        counts.max(),
                        counts.quantile(0.90),
                        counts.quantile(0.95),
                        counts.quantile(0.99),
                    ],
                },
                index=metrics,
            ).astype(float)

        return AnalysisResult(
            raw=stats_df,
            visualizer=self.visualizer.visualize_stats,
        )

    def _context(self, df: pd.DataFrame):
        mask = df.isna()
        n = len(df)
        counts = mask.sum(axis=0).astype(int)

        rates = (
            counts / n
            if n > 0
            else pd.Series(0.0, index=df.columns, dtype=float)
        )
        return counts, rates

        



