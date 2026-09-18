from __future__ import annotations

import matplotlib.pyplot as plt
from ..utils import _title, _style_axes
import pandas as pd
from matplotlib.ticker import PercentFormatter

class RowsMissingnessVisualizer:
    """Visualization accessor for RowsMissingness.

    Usage:
        rm = RowsMissingness(df)
        rm.summary().plot
        rm.stats().plot
        rm.completeness().plot
        rm.missing_above(0.5).plot
    """

    def __init__(self) -> None:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]

    def visualize_full_missing(self, raw: dict, figsize: tuple[float, float] = (6, 4)):
        """Single annotated bar showing count of fully-missing rows."""
        count = raw["count"]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(["Fully missing rows"], [count], color="black")
        ax.set_xlabel("Number of rows", fontsize=11)
        _title(ax, "Fully Missing Rows")
        ax.grid(axis="x", alpha=0.2)

        offset = max(count * 0.02, 0.5)
        ax.text(count + offset, 0, f"{count:,}", va="center", ha="left",
                 fontsize=9, color="black", fontstyle="italic")

        ax.set_xlim(0, max(count * 1.25, 1))
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def visualize_threshold(self, raw: dict, figsize: tuple[float, float] = (6, 4)):
        """Single annotated bar showing count of rows above/below a threshold,
        with the remainder shown for context."""
        count = raw["count"]
        total = raw["total_rows"]
        threshold = raw["threshold"]
        direction = raw["direction"]
        remainder = total - count

        label = f"Rate {'≥' if direction == 'above' else '≤'} {threshold:.1%}"

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh([label, "Remaining rows"], [count, remainder], color="black")
        ax.set_xlabel("Number of rows", fontsize=11)
        _title(ax, "Rows by Missingness Threshold")
        ax.grid(axis="x", alpha=0.2)

        offset = max(total * 0.02, 0.5)
        for i, value in enumerate([count, remainder]):
            ax.text(value + offset, i, f"{value:,}", va="center", ha="left",
                     fontsize=9, color="black", fontstyle="italic")

        ax.set_xlim(0, total * 1.25)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def visualize_completeness(self, raw: pd.DataFrame, figsize: tuple[float, float] = (10, 6)):
        """Bar chart of row counts by number of missing fields."""
        if raw.empty:
            raise ValueError("completeness data is empty")

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(raw["n_missing_fields"], raw["n_rows"], color="black")
        ax.set_xlabel("Number of missing fields", fontsize=11)
        ax.set_ylabel("Number of rows", fontsize=11)
        _title(ax, "Row Completeness")
        ax.grid(axis="y", alpha=0.2)

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def visualize_summary(self, raw: dict, figsize: tuple[float, float] = (8, 5)):
        """Bar chart of summary counts (total, fully missing, threshold counts)."""
        labels = {
            "total_rows": "Total rows",
            "full_missing_rows": "Fully missing",
            "missing_rows_above": "Above threshold",
            "missing_rows_below": "Below threshold",
        }

        items = [(labels[k], v) for k, v in raw.items() if k in labels]
        items = items[::-1]  # so 'Total rows' ends up on top

        names = [i[0] for i in items]
        values = [i[1] for i in items]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(names, values, color="black")
        ax.set_xlabel("Number of rows", fontsize=11)
        _title(ax, "Row Missingness Summary")
        ax.grid(axis="x", alpha=0.2)

        offset = max(max(values) * 0.02, 0.5)
        for i, value in enumerate(values):
            ax.text(value + offset, i, f"{value:,}", va="center", ha="left",
                     fontsize=9, color="black", fontstyle="italic")

        ax.set_xlim(0, max(values) * 1.25)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def visualize_stats(self, raw: pd.Series, figsize: tuple[float, float] = (8, 5)):
        """Horizontal bar chart of row-missingness rate summary statistics."""
        metrics = raw.index[::-1]  # reverse so 'mean' sits at top
        values = raw.reindex(metrics)

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(metrics, values.values, color="black")
        ax.set_xlim(0, max(values.max() * 1.25, 0.01))
        ax.set_xlabel("Rate", fontsize=11)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", alpha=0.2)
        _title(ax, "Row Missingness — Summary Statistics")

        offset = max(values.max() * 0.02, 0.005)
        for i, value in enumerate(values.values):
            ax.text(value + offset, i, f"{value:.1%}", va="center", ha="left",
                     fontsize=9, color="black", fontstyle="italic")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

