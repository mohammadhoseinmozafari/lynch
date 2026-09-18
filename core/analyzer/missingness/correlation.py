
from __future__ import annotations
from typing import Optional
import pandas as pd

from core.analyzer import AnalysisResult, Analyzer 
from scipy import stats
import numpy as np

from core.visualizer.missingness import MissingnessCorrelationVisualizer
class MissingnessCorrelation(Analyzer):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df
        self.mask = df.isna()
        self.visualizer = MissingnessCorrelationVisualizer()

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        return super().analyze(ctx)

    def missing_missing_corr(self) -> AnalysisResult:
        """
        Pairwise correlation between columns' missingness INDICATORS
        (1 = missing, 0 = present). Reveals whether columns go missing
        together (shared cause). Only meaningful for columns that actually
        have missing values; constant columns (all-missing or all-present)
        are dropped to avoid NaN correlations.
        """
        varying = self.mask.loc[:, self.mask.nunique() > 1]

        if varying.shape[1] < 2:
            raw = pd.DataFrame()
        else:
            raw = varying.astype(int).corr().round(3)

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_missing_missing,
        )

    def missing_value_correlation(self) -> AnalysisResult:
        """
        For each column with missing values, tests whether its missingness
        indicator relates to OTHER columns' actual observed values.
        - numeric other column -> point-biserial correlation
        - categorical other column -> Cramer's V (via chi-square)
        This is the more diagnostic signal for MAR: it tells you WHAT the
        missingness depends on, not just which columns co-vary in missingness.
        """
        raw = self._compute_missing_vs_observed(
            missing_cols=[c for c in self.df.columns if self.mask[c].any()],
            other_cols=None,  # all columns except the target
        )

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_missing_observed,
        )

    def missing_target_corr(self, target: str) -> AnalysisResult:
        """
        For each column with missing values (other than the target itself),
        tests whether its missingness indicator relates to the TARGET
        column's observed values. Same statistical logic as
        missing_value_correlation(), scoped to a single column of interest
        (e.g. a label/outcome column) rather than every other column.
        """
        if target not in self.df.columns:
            raise ValueError(f"target column {target!r} not found in dataframe")

        missing_cols = [
            c for c in self.df.columns if c != target and self.mask[c].any()
        ]

        raw = self._compute_missing_vs_observed(
            missing_cols=missing_cols,
            other_cols=[target],
        )

        return AnalysisResult(
            raw=raw,
            visualizer=lambda r: self.visualizer.visualize_missing_target(r, target),
        )

    def _compute_missing_vs_observed(
        self,
        missing_cols: list[str],
        other_cols: Optional[list[str]],
    ) -> pd.DataFrame:
        """Shared logic for missing_value_correlation() and
        missing_target_corr(). other_cols=None means "all columns except
        the one being tested"."""
        results = []

        for target_col in missing_cols:
            indicator = self.mask[target_col].astype(int)

            candidates = (
                other_cols
                if other_cols is not None
                else [c for c in self.df.columns if c != target_col]
            )

            for other_col in candidates:
                other = self.df[other_col]

                valid = other.notna()
                if valid.sum() < 3:
                    continue

                ind_valid = indicator[valid]
                other_valid = other[valid]
                if ind_valid.nunique() < 2:
                    continue

                if pd.api.types.is_numeric_dtype(other_valid):
                    try:
                        corr, p = stats.pointbiserialr(ind_valid, other_valid)
                    except Exception:
                        continue
                    results.append({
                        "missing_column": target_col,
                        "compared_to": other_col,
                        "compared_type": "numeric",
                        "statistic": round(corr, 3),
                        "p_value": round(p, 4),
                        "method": "point-biserial",
                    })
                else:
                    try:
                        contingency = pd.crosstab(ind_valid, other_valid)
                        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                            continue
                        chi2, p, _, _ = stats.chi2_contingency(contingency)
                        n = contingency.values.sum()
                        min_dim = min(contingency.shape) - 1
                        cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan
                    except Exception:
                        continue
                    results.append({
                        "missing_column": target_col,
                        "compared_to": other_col,
                        "compared_type": "categorical",
                        "statistic": round(cramers_v, 3) if not np.isnan(cramers_v) else None,
                        "p_value": round(p, 4),
                        "method": "cramers_v (chi-square)",
                    })

        return pd.DataFrame(results).sort_values("p_value") if results else pd.DataFrame()