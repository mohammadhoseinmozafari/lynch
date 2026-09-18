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

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()

        self.df = df
        self.mask = self.df.isna()

        self.n_cols = self.df.shape[1]
        self.counts = self.mask.sum(axis=1).astype(int)

        self.rates = (
            self.counts / self.n_cols
            if self.n_cols > 0
            else pd.Series(0.0, index=self.df.index, dtype=float)
        )
        self.visualizer = RowsMissingnessVisualizer()

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        pass

    def full_missing(self, sample_size: Optional[int] = None) -> AnalysisResult:
        mask = self.rates == 1.0
        indices = self.rates.index[mask]

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
        threshold: float,
        sample_size: Optional[int] = None,
    ) -> AnalysisResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        mask = self.rates >= threshold
        indices = self.rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        raw = {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
            "threshold": threshold,
            "direction": "above",
            "total_rows": int(len(self.rates)),
        }

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_threshold,
        )

    def missing_below(
        self,
        threshold: float,
        sample_size: Optional[int] = None,
    ) -> AnalysisResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        mask = self.rates <= threshold
        indices = self.rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        raw = {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
            "threshold": threshold,
            "direction": "below",
            "total_rows": int(len(self.rates)),
        }

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_threshold,
        )

    def completeness(self) -> AnalysisResult:
        hist = self.counts.value_counts().sort_index()
        raw = pd.DataFrame({
            "n_missing_fields": hist.index,
            "n_rows": hist.values,
        })

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_completeness,
        )

    def summary(self, thresholds=None) -> AnalysisResult:
        raw = {
            "total_rows": int(len(self.df)),
            "full_missing_rows": int((self.rates == 1.0).sum()),
        }

        if thresholds is not None:
            above = thresholds.get("above")
            if above is not None:
                if not 0.0 <= above <= 1.0:
                    raise ValueError(
                        "thresholds['above'] must be between 0.0 and 1.0"
                    )
                raw["missing_rows_above"] = int((self.rates >= above).sum())

            below = thresholds.get("below")
            if below is not None:
                if not 0.0 <= below <= 1.0:
                    raise ValueError(
                        "thresholds['below'] must be between 0.0 and 1.0"
                    )
                raw["missing_rows_below"] = int((self.rates <= below).sum())

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_summary,
        )

    def stats(self) -> AnalysisResult:
        values = self.rates.dropna().astype(float)

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