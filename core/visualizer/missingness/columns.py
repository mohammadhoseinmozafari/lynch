from __future__ import annotations


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from core.visualizer.utils import _title, _style_axes

class ColumnsMissingnessVisualizer:
    """Visualization accessor for ColumnsMissingness.

    Usage:
        cm = ColumnsMissingness(df)

        cm.plot.missingness()
    """


    def visualize_summary(self , summary, figsize: tuple[float, float] = (10, 6)):
        summary = summary.sort_values("missing_rate", ascending=True)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(summary["column"].values, summary["missing_count"].values, color="black")
        ax.set_xlabel("Missing values", fontsize=11)
        ax.set_ylabel("Column", fontsize=11)
        _title(ax, "Column Missingness")

        ax.grid(axis="x", alpha=0.2)

        offset = max(summary["missing_count"].max() * 0.01, 0.5)
        for i, (missing_count, rate) in enumerate(zip(summary["missing_count"].values, summary["missing_rate"].values)):
            ax.text(
                missing_count + offset, i,
                f"{missing_count:,.0f}  ({rate:.1%})",
                va="center", ha="left",
                fontsize=9, color="black",
                fontstyle="italic",
            )

        # Give room on the right for the annotation text
        ax.set_xlim(0, summary["missing_count"].max() * 1.25)

        _style_axes(fig, ax)
        fig.tight_layout()

        return fig, ax
    
    def visualize_stats(self, stats_df, figsize: tuple[float, float] = (10, 5)):
        """Two-panel horizontal bar chart of summary stats for rates and counts."""
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]

        fig, (ax_rates, ax_counts) = plt.subplots(
            1, 2, figsize=figsize, sharey=True
        )

        metrics = stats_df.index[::-1]  # reverse so 'mean' sits at top
        rates = stats_df["rates"].reindex(metrics)
        counts = stats_df["counts"].reindex(metrics)

        # --- rates panel ---
        ax_rates.barh(metrics, rates.values, color="black")
        ax_rates.set_xlim(0, max(rates.max() * 1.25, 0.01))
        ax_rates.set_xlabel("Rate", fontsize=11)
        ax_rates.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax_rates.grid(axis="x", alpha=0.2)

        offset_r = max(rates.max() * 0.02, 0.005)
        for i, value in enumerate(rates.values):
            ax_rates.text(
                value + offset_r, i, f"{value:.1%}",
                va="center", ha="left", fontsize=9,
                color="black", fontstyle="italic",
            )

        # --- counts panel ---
        ax_counts.barh(metrics, counts.values, color="black")
        ax_counts.set_xlim(0, max(counts.max() * 1.25, 1))
        ax_counts.set_xlabel("Count", fontsize=11)
        ax_counts.grid(axis="x", alpha=0.2)

        offset_c = max(counts.max() * 0.02, 0.5)
        for i, value in enumerate(counts.values):
            ax_counts.text(
                value + offset_c, i, f"{value:,.1f}",
                va="center", ha="left", fontsize=9,
                color="black", fontstyle="italic",
            )

        fig.suptitle("Missingness — Summary Statistics", fontsize=13)

        for ax in (ax_rates, ax_counts):
            _style_axes(fig, ax)

        fig.tight_layout()

        return fig, (ax_rates, ax_counts)

