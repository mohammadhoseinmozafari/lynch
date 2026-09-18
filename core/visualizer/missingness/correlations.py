from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from ..utils import _title, _style_axes
from matplotlib.lines import Line2D


import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---- shared palette / style constants ----
BG = "#fbfaf7"            # warm off-white, classical paper feel
INK = "#2b2620"            # near-black warm ink for text
NUMERIC_CMAP = plt.get_cmap("RdBu_r")   # diverging: blue=negative, red=positive
CAT_CMAP = plt.get_cmap("Oranges")       # sequential, unsigned (Cramer's V)
SIG_EDGE = "#1a1a1a"
NONSIG_EDGE = "#c9c2b8"
GRID_COLOR = "#e4ddd0"


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

    def _style_axes(self, fig, ax):
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#b8afa0")
        ax.tick_params(colors=INK)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)

    def _title(self, ax, text):
        ax.set_title(text, fontsize=14, fontweight="bold", color=INK, pad=14)

    def visualize_missing_missing(self, raw: pd.DataFrame, figsize: tuple[float, float] = (7.5, 6.5)):
        """Diverging heatmap of missingness-indicator correlations between columns."""
        if raw.empty:
            raise ValueError("missing-missing correlation data is empty")

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(raw.values, cmap=NUMERIC_CMAP, vmin=-1, vmax=1)

        ax.set_xticks(range(len(raw.columns)))
        ax.set_xticklabels(raw.columns, rotation=45, ha="right", fontsize=9, color=INK)
        ax.set_yticks(range(len(raw.index)))
        ax.set_yticklabels(raw.index, fontsize=9, color=INK)

        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                value = raw.values[i, j]
                text_color = "white" if abs(value) > 0.55 else INK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color=text_color)

        self._title(ax, "Missingness Correlation (Missing \u2194 Missing)")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        cb.ax.yaxis.label.set_color(INK)
        cb.ax.tick_params(colors=INK)
        fig.patch.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        return fig, ax

    def visualize_missing_observed(
        self,
        raw: pd.DataFrame,
        figsize: tuple[float, float] = (10, 6.2),
        alpha_sig: float = 0.05,
        title: str = "Missingness vs. Observed Values",
    ):
        """
        Colored bubble/dot matrix: missing_column (rows) x compared_to (columns).
        - marker shape: circle = numeric (point-biserial, signed),
                         square = categorical (Cramer's V, unsigned)
        - fill color: diverging red/blue scale (RdBu_r) for numeric sign
                       + strength; sequential orange scale for categorical
                       strength (no sign to encode) -- kept as a SEPARATE
                       colorbar since the two statistics aren't comparable
        - marker size: |statistic| -- larger = stronger association
        - edge weight/color: bold dark = significant (p < alpha_sig),
                              thin warm-gray = not significant
        """
        if raw.empty:
            raise ValueError("correlation data is empty")

        data = raw.dropna(subset=["statistic"])
        if data.empty:
            raise ValueError("no valid statistics to plot")

        missing_order = sorted(data["missing_column"].unique())
        compared_order = sorted(data["compared_to"].unique())
        y_pos = {m: i for i, m in enumerate(missing_order)}
        x_pos = {c: i for i, c in enumerate(compared_order)}

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        max_abs = data["statistic"].abs().max()
        size_scale = 1000 / max(max_abs, 1e-6)
        min_size = 40

        num_norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
        cat_vals = data.loc[data["compared_type"] == "categorical", "statistic"]
        cat_max = cat_vals.max() if not cat_vals.empty else 1.0
        cat_norm = mcolors.Normalize(vmin=0, vmax=cat_max)

        has_numeric = (data["compared_type"] == "numeric").any()
        has_categorical = (data["compared_type"] == "categorical").any()

        for _, row in data.iterrows():
            x, y = x_pos[row["compared_to"]], y_pos[row["missing_column"]]
            stat, p = row["statistic"], row["p_value"]
            is_numeric = row["compared_type"] == "numeric"
            sig = p < alpha_sig
            size = max(abs(stat) * size_scale, min_size)

            if is_numeric:
                marker = "o"
                facecolor = NUMERIC_CMAP(num_norm(stat))
            else:
                marker = "s"
                facecolor = CAT_CMAP(cat_norm(stat))

            edgecolor = SIG_EDGE if sig else NONSIG_EDGE
            linewidth = 2.0 if sig else 1.0

            ax.scatter(x, y, s=size, marker=marker, facecolor=facecolor,
                       edgecolor=edgecolor, linewidth=linewidth, zorder=3)

        ax.set_xticks(range(len(compared_order)))
        ax.set_xticklabels(compared_order, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(missing_order)))
        ax.set_yticklabels([f"{m} missing" for m in missing_order], fontsize=10)
        ax.set_xlim(-0.5, len(compared_order) - 0.5)
        ax.set_ylim(-0.5, len(missing_order) - 0.5)
        ax.invert_yaxis()
        ax.set_xlabel("Observed variable", fontsize=11)
        ax.grid(True, color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)
        self._title(ax, title)

        for spine in ax.spines.values():
            spine.set_visible(False)

        n_bars = int(has_numeric) + int(has_categorical)
        if n_bars == 2:
            cax1 = fig.add_axes([0.93, 0.55, 0.018, 0.28])
            cax2 = fig.add_axes([0.93, 0.15, 0.018, 0.28])
        else:
            cax1 = fig.add_axes([0.93, 0.35, 0.018, 0.4])
            cax2 = None

        if has_numeric:
            sm_num = cm.ScalarMappable(norm=num_norm, cmap=NUMERIC_CMAP)
            sm_num.set_array([])
            cb1 = fig.colorbar(sm_num, cax=cax1)
            cb1.set_label("Numeric (r)", fontsize=8.5, color=INK)
            cb1.ax.tick_params(labelsize=7, colors=INK)
        if has_categorical:
            target_cax = cax2 if cax2 is not None else cax1
            sm_cat = cm.ScalarMappable(norm=cat_norm, cmap=CAT_CMAP)
            sm_cat.set_array([])
            cb2 = fig.colorbar(sm_cat, cax=target_cax)
            cb2.set_label("Categorical (V)", fontsize=8.5, color=INK)
            cb2.ax.tick_params(labelsize=7, colors=INK)

        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#999", markeredgecolor=SIG_EDGE,
                   markersize=10, markeredgewidth=2.0, label="Significant (p < 0.05)", linewidth=0),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#999", markeredgecolor=NONSIG_EDGE,
                   markersize=10, markeredgewidth=1.0, label="Not significant", linewidth=0),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#999", markeredgecolor="#999",
                   markersize=5, label="Weaker", linewidth=0),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#999", markeredgecolor="#999",
                   markersize=13, label="Stronger", linewidth=0),
        ]
        leg = ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.18, 1.0),
                   fontsize=8.5, frameon=False, title="Significance / size", title_fontsize=9,
                   labelspacing=1.3)
        leg.get_title().set_color(INK)
        for text in leg.get_texts():
            text.set_color(INK)

        fig.subplots_adjust(right=0.88)
        return fig, ax

    def visualize_missing_target(
        self,
        raw: pd.DataFrame,
        target: str,
        figsize: tuple[float, float] = (7, 5.5),
        alpha_sig: float = 0.05,
    ):
        """Same colored bubble-matrix encoding as visualize_missing_observed,
        scoped to a single target column."""
        if raw.empty:
            raise ValueError("correlation data is empty")
        return self.visualize_missing_observed(
            raw, figsize=figsize, alpha_sig=alpha_sig,
            title=f"Missingness vs. Target ({target})",
        )