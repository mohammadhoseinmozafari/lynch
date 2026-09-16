from __future__ import annotations

from enum import StrEnum
from typing import  Literal, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from case.domain.value_objects.profile_namespace import ProfileNamespace
from core.visualizer.missingness import MissingnessClustersVisualizer
from .analyzer import Analyzer, AnalyzerType
from scipy import stats

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class MissingnessScope(StrEnum):
    COLUMN = "column"
    ROW = "row"


from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


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

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        pass

    def summary(self) -> pd.DataFrame:
        summary = pd.DataFrame(
            {
                "column": self.df.columns,
                "missing_count": self.counts.values,
                "missing_rate": self.rates.values,
                "dtype": [str(t) for t in self.df.dtypes],
            }
        )

        return (
            summary
            .sort_values("missing_rate", ascending=False)
            .reset_index(drop=True)
        )

    def stats(
        self,
        on: Literal["rates", "counts"] = "rates",
    ) -> dict[str, float]:

        values = self.rates if on == "rates" else self.counts

        if on not in {"rates", "counts"}:
            raise ValueError(
                f"Invalid value for 'on': {on!r}. "
                "Expected 'rates' or 'counts'."
            )

        if values.empty:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
        }


    def visualize(
        self,
        on: Literal["rates", "counts"] = "rates",
        figsize: tuple[float, float] = (10, 6),
    ):
        if on == "rates":
            return self._visualize_rates(figsize)

        if on == "counts":
            return self._visualize_counts(figsize)

        raise ValueError(
            f"Invalid value for 'on': {on!r}. "
            "Expected 'rates' or 'counts'."
        )

    # ------------------------------------------------------------------
    # Rate visualization
    # ------------------------------------------------------------------

    def _visualize_rates(self, figsize):
        rates = (
            self.rates
            .dropna()
            .astype(float)
            .sort_values(ascending=True)
        )

        if rates.empty:
            raise ValueError("missing_rates is empty")

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(
            rates.index,
            rates.values,
            color="black",
        )

        ax.set_xlim(0, 1)

        ax.set_xlabel("Missingness rate")
        ax.set_ylabel("Column")

        ax.set_title(
            "Column Missingness",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        ax.xaxis.set_major_formatter(
            PercentFormatter(1.0)
        )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        ax.set_axisbelow(True)

        # Display percentage values at the end of each bar.
        for i, value in enumerate(rates.values):
            ax.text(
                value + 0.01,
                i,
                f"{value:.1%}",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        fig.tight_layout()

        return fig, ax

    # ------------------------------------------------------------------
    # Count visualization
    # ------------------------------------------------------------------

    def _visualize_counts(self, figsize):
        counts = (
            self.counts
            .dropna()
            .astype(float)
            .sort_values(ascending=True)
        )

        if counts.empty:
            raise ValueError("missing_counts is empty")

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(
            counts.index,
            counts.values,
            color="black",
        )

        ax.set_xlabel("Missing values")
        ax.set_ylabel("Column")

        ax.set_title(
            "Column Missingness Counts",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        ax.set_axisbelow(True)

        # Display count values at the end of each bar.
        offset = max(counts.max() * 0.01, 0.5)

        for i, value in enumerate(counts.values):
            ax.text(
                value + offset,
                i,
                f"{value:,.0f}",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        fig.tight_layout()

        return fig, ax





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

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        pass

    
    def full_missing(self, sample_size: Optional[int] = None):

        mask = self.rates == 1.0

        indices = self.rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        return {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
        }

    def missing_above(
        self,
        threshold: float,
        sample_size: Optional[int] = None,
    ):

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        mask = self.rates >= threshold

        indices = self.rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        return {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
        }

    def missing_below(
        self,
        threshold: float,
        sample_size: Optional[int] = None,
    ):

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        mask = self.rates <= threshold

        indices = self.rates.index[mask]

        if sample_size is not None:
            indices = indices[:sample_size]

        return {
            "count": int(mask.sum()),
            "indices": indices.tolist(),
        }
    
    def completeness(self):
        hist = self.counts.value_counts().sort_index()
        return pd.DataFrame({
            "n_missing_fields": hist.index,
            "n_rows": hist.values,
        })

    def summary(self, thresholds=None):

        summary = {
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

                summary["missing_rows_above"] = int(
                    (self.rates >= above).sum()
                )

            below = thresholds.get("below")

            if below is not None:
                if not 0.0 <= below <= 1.0:
                    raise ValueError(
                        "thresholds['below'] must be between 0.0 and 1.0"
                    )

                summary["missing_rows_below"] = int(
                    (self.rates <= below).sum()
                )

        return summary

    def stats(self):

        values = self.rates.dropna().astype(float)

        if values.empty:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
        }

    def visualize(
    self,
    figsize: tuple[float, float] = (10, 6),
    thresholds: Optional[dict[str, float]] = None,
    ):
        rates = self.rates.dropna().astype(float)

        if rates.empty:
            raise ValueError("missing_rates is empty")

        stats = self.stats()

        fig, ax = plt.subplots(figsize=figsize)

        # 5 percentage-point bins from 0% to 100%
        bins = np.linspace(0.0, 1.0, 21)

        ax.hist(
            rates,
            bins=bins,
            color="black",
            edgecolor="white",
            linewidth=0.8,
        )

        # Mean
        mean = stats["mean"]

        ax.axvline(
            mean,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=f"Mean: {mean:.1%}",
        )

        # Median
        median = stats["median"]

        ax.axvline(
            median,
            color="black",
            linestyle=":",
            linewidth=1.8,
            label=f"Median: {median:.1%}",
        )

        # Optional thresholds
        if thresholds is not None:

            above = thresholds.get("above")

            if above is not None:
                if not 0.0 <= above <= 1.0:
                    raise ValueError(
                        "thresholds['above'] must be between 0.0 and 1.0"
                    )

                ax.axvline(
                    above,
                    color="black",
                    linestyle="-.",
                    linewidth=1.5,
                    label=f"Above: {above:.1%}",
                )

            below = thresholds.get("below")

            if below is not None:
                if not 0.0 <= below <= 1.0:
                    raise ValueError(
                        "thresholds['below'] must be between 0.0 and 1.0"
                    )

                ax.axvline(
                    below,
                    color="black",
                    linestyle="-.",
                    linewidth=1.5,
                    label=f"Below: {below:.1%}",
                )

        ax.set_xlim(0, 1)

        ax.set_xlabel("Missingness rate")
        ax.set_ylabel("Number of rows")

        ax.set_title(
            "Row Missingness Distribution",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        ax.xaxis.set_major_formatter(
            PercentFormatter(1.0)
        )

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        ax.set_axisbelow(True)

        ax.legend()

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        fig.tight_layout()

        return fig, ax

class MissingnessCorrelation(Analyzer):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df
        self.mask = df.isna()

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        return super().analyze(ctx)
    
    def missing_missing_corr(self)-> Optional[pd.DataFrame]:
        """
        Pairwise correlation between columns' missingness INDICATORS
        (1 = missing, 0 = present). Reveals whether columns go missing
        together (shared cause). Only meaningful for columns that actually
        have missing values; constant columns (all-missing or all-present)
        are dropped to avoid NaN correlations.
        """
        varying = self.mask.loc[:, self.mask.nunique() > 1]
        if varying.shape[1] < 2:
            return pd.DataFrame()
        return varying.astype(int).corr().round(3)
    
    def missing_value_correlation(self) -> pd.DataFrame:
        """
        For each column with missing values, tests whether its missingness
        indicator relates to OTHER columns' actual observed values.
        - numeric other column -> point-biserial correlation
        - categorical other column -> Cramer's V (via chi-square)
        This is the more diagnostic signal for MAR: it tells you WHAT the
        missingness depends on, not just which columns co-vary in missingness.
        """
        results = []
        missing_cols = [c for c in self.df.columns if self.mask[c].any()]
        for target_col in missing_cols:
            indicator = self.mask[target_col].astype(int)
            for other_col in self.df.columns:
                if other_col == target_col:
                    continue
                other = self.df[other_col]
                # Only use rows where "other" is observed
                valid = other.notna()
                if valid.sum() < 3:
                    continue
                ind_valid = indicator[valid]
                other_valid = other[valid]
                if ind_valid.nunique() < 2:
                    continue  # missingness doesn't vary among valid rows

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



"""
MissingnessClusters: groups ROWS by their missingness pattern (which
columns are NaN together), as opposed to MissingnessCorrelation, which
looks at pairwise relationships between COLUMNS. Answers questions like:
"is there a distinct 12% of rows that are all missing exactly {income,
score} together?" — i.e. finds the actual missingness clusters, not just
pairwise co-occurrence.
 
Two modes:
- "exact": group rows by their literal missingness signature (which
  columns are NaN). Zero hyperparameters, fully interpretable. Best
  default for datasets with a modest number of columns, where the same
  handful of signatures repeat across many rows.
- "hierarchical": for wider datasets where exact signatures fragment into
  many singleton-ish groups, cluster rows on the binary missingness
  matrix (Hamming/Jaccard distance) via scipy's agglomerative clustering
  and cut into k clusters. Coarser, but keeps things summarizable.
 
Follows the same shape as ColumnsMissingness / RowsMissingness /
MissingnessCorrelation: a pure-data Analyzer class, plus a separate
Plotter class exposed via a lazily-constructed `.plot` property, so
importing/instantiating the analyzer never touches matplotlib.
"""
 
 
 
#  
 
# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------
 
class MissingnessClusters(Analyzer):
    """Cluster rows by their missingness pattern.
 
    Usage:
        mcl = MissingnessClusters(df)
        mcl.clusters()                          # exact signature groups
        mcl.clusters(method="hierarchical", k=6) # coarser, for wide data
        mcl.rows_in(cluster_id=2)                # inspect one cluster
        mcl.plot.sizes()
        mcl.plot.pattern_matrix()
        mcl.plot.dendrogram()
    """
 
    id = "dataset.missingness.clusters"
 
    capability = "missingness.row_clusters"
 
    requires = {"profile.base"}
 
    provides = {capability}
 
    analyzer_type = AnalyzerType.DATASET
 
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.mask = df.isna()
        self._plot: Optional["MissingnessClustersVisualizer"] = None
 
        # only columns that actually vary in missingness carry signal;
        # all-missing / all-present columns would just pad every
        # signature identically and add noise to the distance metric.
        self.varying_cols = list(self.mask.columns[self.mask.nunique() > 1])
 
    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        return super().analyze(ctx)
 
    # ------------------------------------------------------------------
    # Exact-signature clustering
    # ------------------------------------------------------------------
 
    def _exact_clusters(self) -> pd.DataFrame:
        if not self.varying_cols:
            return pd.DataFrame(
                columns=["cluster_id", "size", "rate", "n_missing_cols", "missing_columns"]
            )
 
        sub = self.mask[self.varying_cols]
        # group rows by identical missingness signature
        signatures = sub.apply(tuple, axis=1)
        groups = signatures.value_counts()
 
        rows = []
        for i, (sig, size) in enumerate(groups.items()):
            missing_cols = [c for c, is_missing in zip(self.varying_cols, sig) if is_missing]
            rows.append({
                "cluster_id": i,
                "size": int(size),
                "rate": float(size / len(self.df)),
                "n_missing_cols": len(missing_cols),
                "missing_columns": missing_cols,
            })
 
        out = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
        out["cluster_id"] = range(len(out))  # renumber by size rank
        self._signatures = signatures  # cache for rows_in()
        self._cluster_map = {row["cluster_id"]: i for i, row in enumerate(rows)}
        return out
 
    # ------------------------------------------------------------------
    # Hierarchical clustering (coarser, for wide data)
    # ------------------------------------------------------------------
 
    def _hierarchical_clusters(self, k: int, metric: str = "hamming") -> pd.DataFrame:
        if not self.varying_cols:
            return pd.DataFrame(
                columns=["cluster_id", "size", "rate", "n_missing_cols", "missing_columns"]
            )
 
        sub = self.mask[self.varying_cols].astype(int)
        if len(sub) < 2:
            raise ValueError("Need at least 2 rows to cluster.")
 
        distances = pdist(sub.values, metric=metric)
        Z = linkage(distances, method="average")
        labels = fcluster(Z, t=k, criterion="maxclust")
 
        self._linkage = Z  # cache for dendrogram plot
        self._hier_labels = labels
 
        rows = []
        for cid in sorted(set(labels)):
            row_idx = labels == cid
            size = int(row_idx.sum())
            # columns missing in >50% of the cluster's rows describe it
            cluster_mask = sub.values[row_idx]
            col_rates = cluster_mask.mean(axis=0)
            missing_cols = [c for c, r in zip(self.varying_cols, col_rates) if r > 0.5]
            rows.append({
                "cluster_id": int(cid) - 1,
                "size": size,
                "rate": float(size / len(self.df)),
                "n_missing_cols": len(missing_cols),
                "missing_columns": missing_cols,
            })
 
        return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def clusters(
        self,
        method: Literal["exact", "hierarchical"] = "exact",
        k: int = 8,
        metric: str = "hamming",
    ) -> pd.DataFrame:
        """
        Return one row per missingness cluster: cluster_id, size, rate
        (fraction of all rows), n_missing_cols, and which columns define
        the pattern.
 
        method="exact": rows are grouped by their literal missingness
        signature. No hyperparameters. Can produce many small clusters
        if missingness is scattered/independent across many columns.
 
        method="hierarchical": rows are agglomeratively clustered on
        their binary missingness vectors and cut into `k` clusters.
        Coarser and more stable on wide data, but the reported
        "missing_columns" is a >50%-of-cluster summary, not exact.
        """
        if method == "exact":
            return self._exact_clusters()
        elif method == "hierarchical":
            return self._hierarchical_clusters(k=k, metric=metric)
        else:
            raise ValueError(
                f"Invalid method: {method!r}. Expected 'exact' or 'hierarchical'."
            )
 
    def rows_in(self, cluster_id: int, method: Literal["exact", "hierarchical"] = "exact",
                k: int = 8, metric: str = "hamming") -> pd.DataFrame:
        """Return the subset of the original df belonging to a given
        cluster, so you can inspect why the pattern occurs."""
        if method == "exact":
            if not hasattr(self, "_signatures"):
                self._exact_clusters()
            table = self._exact_clusters()
            if cluster_id not in table["cluster_id"].values:
                raise ValueError(f"No cluster with id {cluster_id}")
            target_sig = tuple(
                col in table.loc[table["cluster_id"] == cluster_id, "missing_columns"].iloc[0]
                for col in self.varying_cols
            )
            row_mask = self._signatures == target_sig
            return self.df.loc[row_mask]
        else:
            self._hierarchical_clusters(k=k, metric=metric)
            row_mask = self._hier_labels == (cluster_id + 1)
            return self.df.loc[row_mask]
 
    def summary(self, method: Literal["exact", "hierarchical"] = "exact", k: int = 8) -> dict:
        table = self.clusters(method=method, k=k)
        if table.empty:
            return {
                "n_clusters": 0,
                "largest_cluster_rate": 0.0,
                "complete_rows_rate": 0.0,
            }
        complete = table.loc[table["n_missing_cols"] == 0, "rate"].sum()
        return {
            "n_clusters": int(len(table)),
            "largest_cluster_rate": float(table["rate"].max()),
            "complete_rows_rate": float(complete),
        }
 
    @property
    def plot(self) -> "MissingnessClustersVisualizer":
        if self._plot is None:
            self._plot = MissingnessClustersVisualizer(self)
        return self._plot
 