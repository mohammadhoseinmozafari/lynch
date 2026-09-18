

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

    def __init__(self, df) -> None:
        super().__init__()

        self.df = df
        self.mask = self.df.isna()

        self.n = len(self.df)
        self.counts = self.mask.sum(axis=0).astype(int)

        self.rates = (
            self.counts / self.n
            if self.n > 0
            else pd.Series(0.0, index=self.df.columns, dtype=float)
        )
        self.visualizer = ColumnsMissingnessVisualizer()

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        pass

    def summary(self) -> AnalysisResult:
        summary = pd.DataFrame(
            {
                "column": self.df.columns,
                "missing_count": self.counts.values,
                "missing_rate": self.rates.values,
                "dtype": [str(t) for t in self.df.dtypes],
            }
        ).sort_values("missing_rate", ascending=False).reset_index(drop=True)
        
        

        return AnalysisResult(
                raw = summary,
                visualizer = self.visualizer.visualize_summary
            )
            
        

    def stats(self) -> AnalysisResult:
        metrics = ["mean", "median", "std", "min", "max", "p90", "p95", "p99"]

        if self.rates.empty:
            stats_df = pd.DataFrame(
                0.0, index=metrics, columns=["rates", "counts"]
            )
        else:
            stats_df = pd.DataFrame(
                {
                    "rates": [
                        self.rates.mean(),
                        self.rates.median(),
                        self.rates.std(),
                        self.rates.min(),
                        self.rates.max(),
                        self.rates.quantile(0.90),
                        self.rates.quantile(0.95),
                        self.rates.quantile(0.99),
                    ],
                    "counts": [
                        self.counts.mean(),
                        self.counts.median(),
                        self.counts.std(),
                        self.counts.min(),
                        self.counts.max(),
                        self.counts.quantile(0.90),
                        self.counts.quantile(0.95),
                        self.counts.quantile(0.99),
                    ],
                },
                index=metrics,
            ).astype(float)

        return AnalysisResult(
            raw=stats_df,
            visualizer=self.visualizer.visualize_stats,
        )




