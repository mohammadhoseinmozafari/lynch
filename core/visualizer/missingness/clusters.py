from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
from scipy.cluster.hierarchy import dendrogram

from core.visualizer.utils import _title, _style_axes


def _set_serif() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]


class MissingnessClustersVisualizer:
    """Visualizations for MissingnessClusters results (ClusterResult)."""

    # ------------------------------------------------------------------
    # Cluster sizes
    # ------------------------------------------------------------------
    def visualize_clusters(self, result, top: int = 15, figsize: tuple[float, float] = (10, 6)):
        """Horizontal bars of cluster sizes with the defining columns annotated."""
        _set_serif()
        table = result.table

        fig, ax = plt.subplots(figsize=figsize)
        if table.empty:
            ax.text(0.5, 0.5, "No varying missingness patterns", ha="center",
                    va="center", style="italic", transform=ax.transAxes)
            ax.set_axis_off()
            _title(ax, "Missingness Clusters")
            return fig, ax

        shown = table.head(top).iloc[::-1]  # largest on top
        labels = [f"#{cid}" for cid in shown["cluster_id"]]

        ax.barh(labels, shown["size"].values, color="black")
        ax.set_xlabel("Rows", fontsize=11)
        ax.set_ylabel("Cluster", fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        _title(ax, "Missingness Clusters")

        offset = max(shown["size"].max() * 0.01, 0.5)
        for i, (_, r) in enumerate(shown.iterrows()):
            cols = r["missing_columns"]
            desc = "complete rows" if not cols else ", ".join(cols[:3]) + (
                f" +{len(cols) - 3}" if len(cols) > 3 else ""
            )
            ax.text(
                r["size"] + offset, i,
                f"{r['size']:,}  ({r['rate']:.1%})  ·  {desc}",
                va="center", ha="left", fontsize=9, color="black", fontstyle="italic",
            )

        ax.set_xlim(0, shown["size"].max() * 1.6)

        if len(table) > top:
            fig.text(0.99, 0.01, f"showing top {top} of {len(table)} clusters",
                     ha="right", va="bottom", fontsize=8, fontstyle="italic", alpha=0.6)

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Pattern matrix
    # ------------------------------------------------------------------
    def visualize_pattern(self, result, top: int = 20, figsize: tuple[float, float] | None = None):
        """Cluster x column matrix. Black = missing, white = present.
        Cell shade shows the fraction of the cluster missing that column."""
        _set_serif()
        table, pattern = result.table, result.pattern

        if table.empty:
            return self.visualize_clusters(result)

        pattern = pattern.head(top)
        sizes = table.set_index("cluster_id").loc[pattern.index, "size"]
        n_rows, n_cols = pattern.shape

        if figsize is None:
            figsize = (max(6, 0.55 * n_cols + 3), max(3, 0.4 * n_rows + 2))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(pattern.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(pattern.columns, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(
            [f"#{cid}  (n={sizes[cid]:,})" for cid in pattern.index], fontsize=9
        )

        # thin cell separators
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

        _title(ax, "Missingness Pattern by Cluster")
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Dendrogram
    # ------------------------------------------------------------------
    def visualize_dendrogram(self, result, figsize: tuple[float, float] = (10, 5)):
        """Dendrogram of unique missingness signatures (hierarchical only)."""
        _set_serif()
        fig, ax = plt.subplots(figsize=figsize)

        if result.linkage is None:
            ax.text(0.5, 0.5, "Dendrogram requires method='hierarchical'",
                    ha="center", va="center", style="italic", transform=ax.transAxes)
            ax.set_axis_off()
            _title(ax, "Missingness Dendrogram")
            return fig, ax

        dendrogram(
            result.linkage, ax=ax, no_labels=True,
            color_threshold=0, above_threshold_color="black",
        )
        ax.set_ylabel(f"Distance ({result.meta.get('metric', 'hamming')})", fontsize=11)
        ax.set_xlabel("Unique missingness signatures", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        _title(ax, "Missingness Dendrogram")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax