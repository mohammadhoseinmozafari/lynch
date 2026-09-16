"""
Plotting layer for ExactDuplicates. Same black-and-white visual language
as the missingness plotters: black bars/markers on a white background,
left-aligned bold titles, light horizontal/vertical gridlines, percentage
axes via PercentFormatter where the value is a rate. No color is used to
carry meaning — conflict/cross-split emphasis is done with fill vs.
hatch/outline-only bars, not color, so the whole module stays visually
consistent under grayscale printing or colorblind-safe review.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.analyzer.duplicates import ExactDuplicates

def _style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=15, fontweight="bold", pad=15)


class ExactDuplicatesVisualizer:
    """Visualization accessor for ExactDuplicates.

    Usage:
        ed = ExactDuplicates(df, split_col="split", label_col="target")
        ed.plot.overview()
        ed.plot.group_sizes()
        ed.plot.rate_by_split()
        ed.plot.label_conflict_breakdown()
    """

    def __init__(self, analyzer: "ExactDuplicates") -> None:
        self._a = analyzer

    def overview(self, figsize: tuple[float, float] = (7, 5)):
        """Single bar: overall exact duplicate rate, with the raw count
        annotated. The first, cheapest thing to look at per the spec —
        severity at a glance before investigating further."""
        summary = self._a.duplicate_rate()

        fig, ax = plt.subplots(figsize=figsize)
        rate = summary["rate"]
        ax.bar(["Duplicate rows"], [rate], color="black", width=0.1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of rows")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        _title(ax, "Exact Duplicate Rate")

        ax.text(
            0, rate + max(rate * 0.03, 0.003),
            f"{rate:.2%}  ({summary['n_duplicate_rows']:,} rows, "
            f"{summary['n_groups']:,} groups)",
            ha="center", va="bottom", fontsize=9, color="black",
        )

        ax.grid(axis="y", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def group_sizes(self, top_n: int = 20, figsize: tuple[float, float] = (9, 6)):
        """Horizontal bar chart of the largest duplicate groups, by
        number of rows in the group. Surfaces whether duplication is a
        few rows copied many times (one very large bar) vs. many
        independent pairs (many bars of size 2)."""
        table = self._a.duplicate_groups()
        if table.empty:
            raise ValueError("No duplicate groups to plot.")

        top = table.head(top_n).iloc[::-1]
        labels = [f"group {gid}" for gid in top["group_id"]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, top["size"], color="black")
        ax.set_xlabel("Rows in group")
        _title(ax, "Largest Duplicate Groups")


        ax.grid(axis="x", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def rate_by_split(self, figsize: tuple[float, float] = (8, 6)):
        """Bar chart of duplicate group counts by split involvement:
        within one split vs. spanning multiple splits (cross-split).
        Cross-split bars are drawn hatched rather than colored, so the
        distinction reads clearly without introducing color. Requires
        split_col."""
        table = self._a.cross_split_overlap()
        if table.empty:
            raise ValueError("No duplicate groups to plot.")

        within = int((~table["is_cross_split"]).sum())
        cross = int(table["is_cross_split"].sum())

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            ["Within-split", "Cross-split"],
            [within, cross],
            color=["black", "white"],
            edgecolor="black",
            linewidth=1.5,
        )
        bars[1].set_hatch("///")

        ax.set_ylabel("Number of duplicate groups")
        _title(ax, "Duplicate Groups: Within-Split vs. Cross-Split")

        for bar, count in zip(bars, [within, cross]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{count:,}", ha="center", va="bottom", fontsize=10, color="black")

        ax.grid(axis="y", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def label_conflict_breakdown(self, figsize: tuple[float, float] = (8, 6)):
        """Bar chart of duplicate groups with consistent vs. conflicting
        labels. Conflicting groups are drawn hatched, matching
        rate_by_split()'s visual convention for "the bad outcome".
        Requires label_col."""
        table = self._a.label_conflicts()
        if table.empty:
            raise ValueError("No duplicate groups to plot.")

        consistent = int((~table["has_conflict"]).sum())
        conflicting = int(table["has_conflict"].sum())

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            ["Consistent labels", "Conflicting labels"],
            [consistent, conflicting],
            color=["black", "white"],
            edgecolor="black",
            linewidth=1.5,
        )
        bars[1].set_hatch("///")

        ax.set_ylabel("Number of duplicate groups")
        _title(ax, "Duplicate Groups: Label Consistency")

        for bar, count in zip(bars, [consistent, conflicting]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{count:,}", ha="center", va="bottom", fontsize=10, color="black")

        ax.grid(axis="y", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax