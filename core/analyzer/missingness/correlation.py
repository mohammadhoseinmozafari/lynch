
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
    

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


class VectorizedMissingnessCorrelation(Analyzer):
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_sample_size: Optional[int] = 20_000,
        min_valid: int = 3,
    ) -> None:
        """
        categorical_sample_size: for the Cramer's V path, rows are
            subsampled to this size when the dataset is larger, since
            chi-square significance stabilizes well before using every
            row and this is the dominant remaining cost at scale.
            Pass None to disable sampling and always use the full data.
        min_valid: minimum number of jointly-valid rows required for a
            pair to be tested at all.
        """
        super().__init__()
        self.df = df
        self.mask = df.isna()
        self.categorical_sample_size = categorical_sample_size
        self.min_valid = min_valid
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
            # np.corrcoef on the full indicator matrix -- no missing data
            # here (mask is always complete), so this is a single call.
            arr = varying.astype(float).values
            corr = np.corrcoef(arr, rowvar=False)
            raw = pd.DataFrame(corr, index=varying.columns, columns=varying.columns).round(3)

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_missing_missing,
        )

    def missing_value_correlation(self) -> AnalysisResult:
        """
        For each column with missing values, tests whether its missingness
        indicator relates to OTHER columns' actual observed values.
        - numeric other column -> point-biserial correlation (vectorized)
        - categorical other column -> Cramer's V (bincount + optional
          row-sampling for large data)
        """
        missing_cols = [c for c in self.df.columns if self.mask[c].any()]

        raw = self._compute_missing_vs_observed(
            missing_cols=missing_cols,
            other_cols=list(self.df.columns),
        )

        return AnalysisResult(
            raw=raw,
            visualizer=self.visualizer.visualize_missing_observed,
        )

    def missing_target_corr(self, target: str) -> AnalysisResult:
        """
        For each column with missing values (other than the target itself),
        tests whether its missingness indicator relates to the TARGET
        column's observed values. Same logic as missing_value_correlation(),
        scoped to a single column of interest.
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

    # ------------------------------------------------------------------
    # Vectorized computation
    # ------------------------------------------------------------------

    def _compute_missing_vs_observed(
        self,
        missing_cols: list[str],
        other_cols: list[str],
    ) -> pd.DataFrame:
        if not missing_cols:
            return pd.DataFrame()

        indicator_df = self.mask[missing_cols].astype(int)

        results = []

        for target_col in missing_cols:
            candidates = [c for c in other_cols if c != target_col]
            numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(self.df[c])]
            cat_cols = [c for c in candidates if c not in numeric_cols]

            single_indicator = indicator_df[[target_col]]

            if numeric_cols:
                corr_df, p_df, n_df = self._vectorized_pointbiserial(
                    single_indicator, self.df[numeric_cols]
                )
                for o_col in corr_df.columns:
                    stat = corr_df.loc[target_col, o_col]
                    if pd.isna(stat):
                        continue
                    results.append({
                        "missing_column": target_col,
                        "compared_to": o_col,
                        "compared_type": "numeric",
                        "statistic": round(float(stat), 3),
                        "p_value": round(float(p_df.loc[target_col, o_col]), 4),
                        "method": "point-biserial",
                    })

            if cat_cols:
                v_df, p_df = self._vectorized_cramers_v(
                    single_indicator, self.df[cat_cols]
                )
                for o_col in v_df.columns:
                    stat = v_df.loc[target_col, o_col]
                    if pd.isna(stat):
                        continue
                    results.append({
                        "missing_column": target_col,
                        "compared_to": o_col,
                        "compared_type": "categorical",
                        "statistic": round(float(stat), 3),
                        "p_value": round(float(p_df.loc[target_col, o_col]), 4),
                        "method": "cramers_v (chi-square)",
                    })

        return pd.DataFrame(results).sort_values("p_value") if results else pd.DataFrame()

    def _vectorized_pointbiserial(
        self, indicator_df: pd.DataFrame, numeric_df: pd.DataFrame
    ):
        """Vectorized point-biserial correlation between every indicator
        column and every numeric column, exact match to
        scipy.stats.pointbiserialr, handling per-pair NaNs via masked
        sum-based correlation formula."""
        if indicator_df.empty or numeric_df.empty:
            empty = pd.DataFrame(index=indicator_df.columns, columns=numeric_df.columns, dtype=float)
            return empty, empty.copy(), empty.copy()

        M = indicator_df.values.astype(np.float64)
        V = numeric_df.values.astype(np.float64)
        valid = ~np.isnan(V)
        Vf = np.where(valid, V, 0.0)
        validf = valid.astype(np.float64)

        k, m = M.shape[1], V.shape[1]

        n_valid = validf.sum(axis=0)
        sum_V = (Vf * validf).sum(axis=0)
        sum_V2 = (Vf**2 * validf).sum(axis=0)

        sum_M = M.T @ validf
        sum_M2 = sum_M  # binary indicator: M^2 == M
        sum_MV = M.T @ (Vf * validf)

        n_b = np.broadcast_to(n_valid, (k, m))
        sum_V_b = np.broadcast_to(sum_V, (k, m))
        sum_V2_b = np.broadcast_to(sum_V2, (k, m))

        num = n_b * sum_MV - sum_M * sum_V_b
        den = np.sqrt((n_b * sum_M2 - sum_M**2) * (n_b * sum_V2_b - sum_V_b**2))

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = num / den
        corr[den == 0] = np.nan
        corr[n_b < self.min_valid] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            t_stat = corr * np.sqrt((n_b - 2) / (1 - corr**2))
        p_value = 2 * stats.t.sf(np.abs(t_stat), n_b - 2)
        p_value[np.isnan(corr)] = np.nan

        corr_df = pd.DataFrame(corr, index=indicator_df.columns, columns=numeric_df.columns)
        p_df = pd.DataFrame(p_value, index=indicator_df.columns, columns=numeric_df.columns)
        n_df = pd.DataFrame(n_b.astype(int), index=indicator_df.columns, columns=numeric_df.columns)
        return corr_df, p_df, n_df

    def _vectorized_cramers_v(
        self, indicator_df: pd.DataFrame, cat_df: pd.DataFrame
    ):
        """Cramer's V between every indicator column and every categorical
        column. Contingency tables are built with np.bincount (fast) rather
        than pd.crosstab; category codes are factorized once per column
        rather than per pair. Rows are subsampled to
        self.categorical_sample_size when the data is larger, since the
        chi-square statistic's significance stabilizes well before all
        rows are used -- this is the dominant remaining cost since
        chi2_contingency itself is still called once per pair."""
        if indicator_df.empty or cat_df.empty:
            empty = pd.DataFrame(index=indicator_df.columns, columns=cat_df.columns, dtype=float)
            return empty, empty.copy()

        if (
            self.categorical_sample_size is not None
            and len(indicator_df) > self.categorical_sample_size
        ):
            rng = np.random.default_rng(0)
            idx = rng.choice(len(indicator_df), size=self.categorical_sample_size, replace=False)
            indicator_df = indicator_df.iloc[idx]
            cat_df = cat_df.iloc[idx]

        codes_cache, ncats_cache, valid_cache = {}, {}, {}
        for col in cat_df.columns:
            codes, uniques = pd.factorize(cat_df[col], use_na_sentinel=True)
            codes_cache[col] = codes
            ncats_cache[col] = len(uniques)
            valid_cache[col] = codes != -1

        v_df = pd.DataFrame(index=indicator_df.columns, columns=cat_df.columns, dtype=float)
        p_df = pd.DataFrame(index=indicator_df.columns, columns=cat_df.columns, dtype=float)

        for i_col in indicator_df.columns:
            ind_full = indicator_df[i_col].values
            for c_col in cat_df.columns:
                valid = valid_cache[c_col]
                if valid.sum() < self.min_valid:
                    continue
                ind_valid = ind_full[valid]
                if len(np.unique(ind_valid)) < 2:
                    continue
                codes_valid = codes_cache[c_col][valid]
                v, p = self._cramers_v_pair(ind_valid, codes_valid, ncats_cache[c_col])
                v_df.loc[i_col, c_col] = v
                p_df.loc[i_col, c_col] = p

        return v_df, p_df

    def _cramers_v_pair(self, indicator: np.ndarray, codes: np.ndarray, n_categories: int):
        n = len(indicator)
        if n < self.min_valid:
            return np.nan, np.nan
        combined = indicator * n_categories + codes
        table = np.bincount(combined, minlength=2 * n_categories).reshape(2, n_categories)
        col_sums = table.sum(axis=0)
        table = table[:, col_sums > 0]
        if table.shape[1] < 2 or table.shape[0] < 2:
            return np.nan, np.nan
        row_sums = table.sum(axis=1)
        if (row_sums == 0).any():
            return np.nan, np.nan
        try:
            chi2, p, _, _ = stats.chi2_contingency(table)
        except Exception:
            return np.nan, np.nan
        min_dim = min(table.shape) - 1
        if min_dim <= 0:
            return np.nan, np.nan
        return np.sqrt((chi2 / n) / min_dim), p