from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from scipy.cluster.hierarchy import dendrogram
from core.analyzer.missingness import MissingnessClusters
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