from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from ..utils import _title, _style_axes
class MissingnessCorrelationVisualizer:
    """Visualization accessor for MissingnessCorrelation.

    Usage:
        mc = MissingnessCorrelation(df)
        mc.missing_missing_corr().plot
        mc.missing_value_correlation().plot
        mc.missing_target_corr("label").plot
    """

    def __init__(self) -> None:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]

    def visualize_missing_missing(self, raw: pd.DataFrame, figsize: tuple[float, float] = (8, 7)):
        """Heatmap of missingness-indicator correlations between columns."""
        if raw.empty:
            raise ValueError("missing-missing correlation data is empty")

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(raw.values, cmap="Greys", vmin=-1, vmax=1)

        ax.set_xticks(range(len(raw.columns)))
        ax.set_xticklabels(raw.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(raw.index)))
        ax.set_yticklabels(raw.index, fontsize=9)

        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                value = raw.values[i, j]
                text_color = "white" if abs(value) > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color=text_color)

        _title(ax, "Missingness Correlation (Missing \u2194 Missing)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

        fig.tight_layout()
        return fig, ax

    def visualize_missing_observed(self, raw: pd.DataFrame, figsize: tuple[float, float] = (10, 6)):
        """Horizontal bar chart of the strongest missing-vs-observed
        associations, ranked by |statistic|, annotated with p-values."""
        if raw.empty:
            raise ValueError("missing-observed correlation data is empty")

        return self._bar_by_statistic(
            raw, title="Missingness vs. Observed Values", figsize=figsize
        )

    def visualize_missing_target(
        self,
        raw: pd.DataFrame,
        target: str,
        figsize: tuple[float, float] = (10, 6),
    ):
        """Horizontal bar chart of each column's missingness association
        with the target column, ranked by |statistic|."""
        if raw.empty:
            raise ValueError("missing-target correlation data is empty")

        return self._bar_by_statistic(
            raw,
            title=f"Missingness vs. Target ({target})",
            figsize=figsize,
            label_col="missing_column",
        )

    def _bar_by_statistic(
        self,
        raw: pd.DataFrame,
        title: str,
        figsize: tuple[float, float],
        label_col: Optional[str] = None,
        top_n: int = 20,
    ):
        """Shared rendering for missing_value_correlation and
        missing_target_corr results: bars sized by |statistic|, sorted
        ascending (so strongest sits at top), annotated with p-values."""
        data = raw.dropna(subset=["statistic"]).copy()
        if data.empty:
            raise ValueError("no valid statistics to plot")

        data["abs_stat"] = data["statistic"].abs()
        data = data.sort_values("abs_stat", ascending=True).tail(top_n)

        if label_col is not None:
            labels = data[label_col]
        else:
            labels = data["missing_column"] + " \u2192 " + data["compared_to"]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, data["statistic"].values, color="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Association strength (correlation / Cramer's V)", fontsize=11)
        _title(ax, title)
        ax.grid(axis="x", alpha=0.2)

        offset = max(data["statistic"].abs().max() * 0.03, 0.01)
        for i, (stat, p) in enumerate(zip(data["statistic"].values, data["p_value"].values)):
            ha = "left" if stat >= 0 else "right"
            x = stat + offset if stat >= 0 else stat - offset
            ax.text(x, i, f"{stat:.2f} (p={p:.3f})", va="center", ha=ha,
                     fontsize=8, color="black", fontstyle="italic")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax