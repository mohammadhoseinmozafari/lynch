"""
Plotting layer for the missingness analyzers.

Design:
- Each Analyzer (ColumnsMissingness, RowsMissingness, MissingnessCorrelation)
  gets a small, dedicated Visualizer class that takes the analyzer instance in
  its constructor and calls back into the analyzer's *data* methods
  (summary(), stats(), missing_missing_corr(), ...). Visualizers never compute
  statistics themselves — they only shape/style figures from data the
  analyzer already exposes. That keeps a single source of truth.
- Each analyzer exposes the Visualizer via a `.plot` property, constructed
  lazily so importing/instantiating the analyzer never touches matplotlib
  until `.plot.<something>()` is actually called.
- Every plot method returns (fig, ax) so callers can keep customizing
  or saving the figure, consistent with the existing `visualize()` methods.
- Nothing here changes existing return types of analyze()/summary()/stats()/
  missing_missing_corr()/missing_value_correlation(). Existing callers of
  ColumnsMissingness.visualize() / RowsMissingness.visualize() keep working
  unchanged (see the thin backward-compat shims at the bottom).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from scipy.cluster.hierarchy import dendrogram
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.analyzer.missingness import (
        ColumnsMissingness,
        RowsMissingness,
        MissingnessCorrelation,
        MissingnessClusters,
    )
# ----------------------------------------------------------------------
# Shared styling helpers (kept tiny + local so Visualizers stay readable)
# ----------------------------------------------------------------------

def _style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=15, fontweight="bold", pad=15)


# ----------------------------------------------------------------------
# ColumnsMissingness
# ----------------------------------------------------------------------

class ColumnsMissingnessVisualizer:
    """Visualization accessor for ColumnsMissingness.

    Usage:
        cm = ColumnsMissingness(df)
        cm.plot.rates()
        cm.plot.counts()
    """

    def __init__(self, analyzer: ColumnsMissingness) -> None:
        self._a = analyzer

    def rates(self, figsize: tuple[float, float] = (10, 6)):
        """Horizontal bar chart of per-column missingness rate."""
        rates = self._a.rates.dropna().astype(float).sort_values(ascending=True)
        if rates.empty:
            raise ValueError("missing_rates is empty")

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(rates.index, rates.values, color="black")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Missingness rate")
        ax.set_ylabel("Column")
        _title(ax, "Column Missingness")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.2)

        for i, value in enumerate(rates.values):
            ax.text(value + 0.01, i, f"{value:.1%}", va="center", ha="left",
                     fontsize=9, color="black")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def counts(self, figsize: tuple[float, float] = (10, 6)):
        """Horizontal bar chart of per-column missing value counts."""
        counts = self._a.counts.dropna().astype(float).sort_values(ascending=True)
        if counts.empty:
            raise ValueError("missing_counts is empty")

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(counts.index, counts.values, color="black")
        ax.set_xlabel("Missing values")
        ax.set_ylabel("Column")
        _title(ax, "Column Missingness Counts")
        ax.grid(axis="x", alpha=0.2)

        offset = max(counts.max() * 0.01, 0.5)
        for i, value in enumerate(counts.values):
            ax.text(value + offset, i, f"{value:,.0f}", va="center", ha="left",
                     fontsize=9, color="black")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# RowsMissingness
# ----------------------------------------------------------------------

class RowsMissingnessVisualizer:
    """Visualization accessor for RowsMissingness.

    Usage:
        rm = RowsMissingness(df)
        rm.plot.distribution()
        rm.plot.distribution(thresholds={"above": 0.5})
        rm.plot.completeness()
    """

    def __init__(self, analyzer: RowsMissingness) -> None:
        self._a = analyzer

    def distribution(
        self,
        figsize: tuple[float, float] = (10, 6),
        thresholds: Optional[dict[str, float]] = None,
    ):
        """Histogram of per-row missingness rate, with mean/median lines
        and optional threshold markers."""
        rates = self._a.rates.dropna().astype(float)
        if rates.empty:
            raise ValueError("missing_rates is empty")

        s = self._a.stats()

        fig, ax = plt.subplots(figsize=figsize)
        bins = np.linspace(0.0, 1.0, 21)
        ax.hist(rates, bins=bins, color="black", edgecolor="white", linewidth=0.8)

        ax.axvline(s["mean"], color="black", linestyle="--", linewidth=1.8,
                   label=f"Mean: {s['mean']:.1%}")
        ax.axvline(s["median"], color="black", linestyle=":", linewidth=1.8,
                   label=f"Median: {s['median']:.1%}")

        if thresholds is not None:
            above = thresholds.get("above")
            if above is not None:
                if not 0.0 <= above <= 1.0:
                    raise ValueError("thresholds['above'] must be between 0.0 and 1.0")
                ax.axvline(above, color="black", linestyle="-.", linewidth=1.5,
                           label=f"Above: {above:.1%}")

            below = thresholds.get("below")
            if below is not None:
                if not 0.0 <= below <= 1.0:
                    raise ValueError("thresholds['below'] must be between 0.0 and 1.0")
                ax.axvline(below, color="black", linestyle="-.", linewidth=1.5,
                           label=f"Below: {below:.1%}")

        ax.set_xlim(0, 1)
        ax.set_xlabel("Missingness rate")
        ax.set_ylabel("Number of rows")
        _title(ax, "Row Missingness Distribution")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.2)
        ax.legend()

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def completeness(self, figsize: tuple[float, float] = (10, 6)):
        """Bar chart of row counts by number of missing fields, sourced
        from RowsMissingness.completeness()."""
        hist = self._a.completeness()
        if hist.empty:
            raise ValueError("completeness() returned no data")

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(hist["n_missing_fields"], hist["n_rows"], color="black")
        ax.set_xlabel("Number of missing fields")
        ax.set_ylabel("Number of rows")
        _title(ax, "Row Completeness")
        ax.grid(axis="y", alpha=0.2)

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax


# ----------------------------------------------------------------------
# MissingnessCorrelation
# ----------------------------------------------------------------------

class MissingnessCorrelationVisualizer:
    """Visualization accessor for MissingnessCorrelation.

    Usage:
        mc = MissingnessCorrelation(df)
        mc.plot.heatmap()
        mc.plot.dependence_bar(top_n=10)
    """

    def __init__(self, analyzer: MissingnessCorrelation) -> None:
        self._a = analyzer

    def heatmap(self, figsize: tuple[float, float] = (8, 8), annot: bool = True):
        """Heatmap of missing_missing_corr() — which columns go missing
        together. Uses matplotlib only (no seaborn dependency)."""
        corr = self._a.missing_missing_corr()
        if corr is None or corr.empty:
            raise ValueError("No varying missing columns to plot.")

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)

        if annot:
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    val = corr.values[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=8,
                             color="white" if abs(val) > 0.5 else "black")

        _title(ax, "Missingness Correlation (co-occurrence)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")

        fig.patch.set_facecolor("white")
        fig.tight_layout()
        return fig, ax

    def dependence_bar(
        self,
        top_n: int = 15,
        alpha: float = 0.05,
        figsize: tuple[float, float] = (9, 6),
    ):
        """Bar chart of the strongest missingness-vs-observed-value
        associations from missing_value_correlation(). Bars are colored by
        whether p_value < alpha; magnitude comes from `statistic`
        (point-biserial r or Cramer's V, both roughly comparable in [-1, 1]
        or [0, 1] respectively)."""
        res = self._a.missing_value_correlation()
        if res is None or res.empty:
            raise ValueError("No dependence results to plot.")

        res = res.dropna(subset=["statistic"]).copy()
        if res.empty:
            raise ValueError("No non-null statistics to plot.")

        res["abs_stat"] = res["statistic"].abs()
        top = res.sort_values("abs_stat", ascending=False).head(top_n)
        top = top.iloc[::-1]  # so the strongest ends up at the top of barh

        labels = top["missing_column"] + " ~ " + top["compared_to"]
        colors = np.where(top["p_value"] < alpha, "black", "lightgray")

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, top["statistic"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Association strength (r / Cramer's V)")
        _title(ax, "Strongest Missingness Dependencies")

        # Legend proxy for significance coloring
        from matplotlib.patches import Patch
        handles = [
            Patch(color="black", label=f"p < {alpha}"),
            Patch(color="lightgray", label=f"p >= {alpha}"),
        ]
        ax.legend(handles=handles, loc="lower right")

        ax.grid(axis="x", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax


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


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



# ----------------------------------------------------------------------
# Plotter
# ----------------------------------------------------------------------

class MissingnessClustersVisualizer:
    """Visualization accessor for MissingnessClusters.

    Usage:
        mcl = MissingnessClusters(df)
        mcl.plot.sizes()
        mcl.plot.pattern_matrix(top_n=10)
        mcl.plot.dendrogram()   # requires method="hierarchical" to have run
    """

    def __init__(self, analyzer: "MissingnessClusters") -> None:
        self._a = analyzer

    def sizes(
        self,
        method: Literal["exact", "hierarchical"] = "exact",
        k: int = 8,
        top_n: int = 15,
        figsize: tuple[float, float] = (9, 6),
    ):
        """Horizontal bar chart of cluster sizes (fraction of rows),
        labeled by which columns are missing in that cluster."""
        table = self._a.clusters(method=method, k=k)
        if table.empty:
            raise ValueError("No missingness clusters to plot.")

        top = table.head(top_n).iloc[::-1]
        labels = [
            "complete" if not cols else " + ".join(cols)
            for cols in top["missing_columns"]
        ]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, top["rate"], color="black")
        ax.set_xlim(0, max(top["rate"].max() * 1.15, 0.05))
        ax.set_xlabel("Share of rows")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title("Missingness Clusters", loc="left", fontsize=15,
                      fontweight="bold", pad=15)

        for i, (rate, size) in enumerate(zip(top["rate"], top["size"])):
            ax.text(rate + max(top["rate"].max() * 0.01, 0.002), i,
                     f"{rate:.1%} (n={size:,})", va="center", ha="left",
                     fontsize=9, color="black")

        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.tight_layout()
        return fig, ax

    def pattern_matrix(
        self,
        method: Literal["exact", "hierarchical"] = "exact",
        k: int = 8,
        top_n: int = 15,
        figsize: tuple[float, float] = (9, 7),
    ):
        """Grid: rows = top clusters (ordered by size), columns = dataset
        columns, cell filled if that column is missing in that cluster's
        pattern. Makes shared-cause patterns visually obvious at a glance."""
        table = self._a.clusters(method=method, k=k)
        if table.empty:
            raise ValueError("No missingness clusters to plot.")

        top = table.head(top_n)
        cols = self._a.varying_cols
        grid = np.zeros((len(top), len(cols)))
        for i, missing_cols in enumerate(top["missing_columns"]):
            for j, c in enumerate(cols):
                grid[i, j] = 1 if c in missing_cols else 0

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(grid, cmap="Greys", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f"{r:.1%} (n={s:,})" for r, s in zip(top["rate"], top["size"])])
        ax.set_xlabel("Column")
        ax.set_ylabel("Cluster (share of rows)")
        ax.set_title("Missingness Pattern by Cluster", loc="left", fontsize=15,
                      fontweight="bold", pad=15)

        # gridlines between cells
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(top), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

        fig.patch.set_facecolor("white")
        fig.tight_layout()
        return fig, ax

    def dendrogram(
        self,
        k: int = 8,
        metric: str = "hamming",
        figsize: tuple[float, float] = (10, 6),
        truncate: Optional[int] = 30,
    ):
        """Dendrogram of row-wise hierarchical clustering on missingness
        vectors. Running this triggers method="hierarchical" clustering
        as a side effect (needed to compute the linkage matrix)."""
        self._a._hierarchical_clusters(k=k, metric=metric)
        Z = self._a._linkage

        fig, ax = plt.subplots(figsize=figsize)
        kwargs = {"truncate_mode": "lastp", "p": truncate} if truncate else {}
        dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color="black", **kwargs)
        ax.set_title("Row Missingness Dendrogram", loc="left", fontsize=15,
                      fontweight="bold", pad=15)
        ax.set_xlabel("Rows (or row groups)")
        ax.set_ylabel("Distance")
        ax.set_xticklabels([])

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        fig.tight_layout()
        return fig, ax