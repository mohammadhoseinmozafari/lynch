from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from .utils import _style_axes, _title
from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from core.analyzer.coverage.density import DensityEstimate
    

class DensityEstimateVisualizer:
    """Visualization accessor for DensityEstimate.

    Usage:
        de = DensityEstimate(df, columns=["age", "income", "tenure"])
        de.plot.overview()
        de.plot.density_1d("income")
        de.plot.density_2d(("age", "income"))
        de.plot.density_projection(method="pca")
    """

    def __init__(self, analyzer: "DensityEstimate") -> None:
        self._a = analyzer

    def overview(self, figsize: tuple[float, float] = (7, 5)):
        """Single bar: share of training rows in the low-density
        ("hole") region, at the auto threshold. Cheapest first look,
        matching ExactDuplicates.plot.overview()."""
        summary = self._a.density_summary()

        fig, ax = plt.subplots(figsize=figsize)
        rate = summary["pct_rows_low_density"]
        ax.bar(["Low-density rows"], [rate], color="black", width=0.1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share of rows")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        _title(ax, "Low-Density (Sparse) Training Rows")

        ax.text(
            0, rate + max(rate * 0.03, 0.003),
            f"{rate:.2%}  (threshold: {summary['low_density_threshold']:.2f} "
            f"log-density, {summary['method']})",
            ha="center", va="bottom", fontsize=9, color="black",
        )

        ax.grid(axis="y", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def density_1d(self, feature: str, figsize: tuple[float, float] = (8, 5)):
        """Line plot of the marginal density along a single feature,
        holding other dims at their mean. Simple sanity-check view
        before reaching for the 2D/projection plots."""
        if feature not in self._a.columns:
            raise ValueError(f"{feature!r} is not a column this estimator was fit on.")

        vals = self._a.df[feature].values
        pad = (vals.max() - vals.min()) * 0.05
        grid = np.linspace(vals.min() - pad, vals.max() + pad, 300)

        query = pd.DataFrame({feature: grid})
        for c in self._a.columns:
            if c != feature:
                query[c] = self._a.df[c].mean()
        query = query[self._a.columns]

        log_density = self._a.score(query)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(grid, np.exp(log_density), color="black", linewidth=1.8)
        ax.fill_between(grid, np.exp(log_density), color="black", alpha=0.08)

        # rug of actual training points along the x-axis
        ax.plot(
            vals, np.zeros_like(vals) - 0.02 * np.exp(log_density).max(),
            "|", color="black", alpha=0.3, markersize=8, clip_on=False,
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        _title(ax, f"Marginal Density — {feature}")

        ax.grid(axis="y", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def density_2d(
        self,
        dims: tuple[str, str],
        resolution: int = 100,
        figsize: tuple[float, float] = (8, 7),
        show_points: bool = True,
        max_points: int = 2000,
    ):
        """Filled contour heatmap of density over two features, with
        actual training points overlaid as a scatter. The primary
        visualization for "where is data dense/sparse" in 2D — and the
        same grid primitive GapDetector reuses for train-vs-reference
        comparison plots.

        Grayscale colormap to match the minimalist black/white/hatch
        convention used elsewhere — darker = denser.
        """
        grid = self._a.density_grid(dims, resolution=resolution)
        x_col, y_col = dims

        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(
            grid["xx"], grid["yy"], grid["density"],
            levels=15, cmap="Greys",
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Log-density", color="black")

        if show_points:
            pts = self._a.df[[x_col, y_col]]
            if len(pts) > max_points:
                pts = pts.sample(n=max_points, random_state=0)
            ax.scatter(
                pts[x_col], pts[y_col],
                s=6, color="white", edgecolor="black", linewidth=0.3, alpha=0.6,
            )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_xlim(*grid["x_edges"])
        ax.set_ylim(*grid["y_edges"])
        _title(ax, f"Density — {x_col} vs {y_col}")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def density_projection(
        self,
        method: Literal["pca"] = "pca",
        resolution: int = 100,
        figsize: tuple[float, float] = (8, 7),
        max_points: int = 2000,
    ):
        """For >2 dimensions: project training data to its top-2
        principal components, refit a lightweight 2D KDE on the
        projected points (for display purposes only — this is NOT the
        same as slicing self._a's actual fitted density, since PCA axes
        aren't members of self.columns), and render the same contour +
        scatter view as density_2d().

        This trades fidelity for visualizability: a hole in 2D PCA
        space is not guaranteed to be a hole in the original space, and
        vice versa. Use it as an orientation view, and drill into
        specific density_2d(dims) pairs — or low_density_regions()'s
        actual centroids — for anything you intend to act on.
        """
        if method != "pca":
            raise ValueError("Only method='pca' is currently supported.")

        X = self._a._scaler.transform(self._a.df[self._a.columns].values)
        pca = PCA(n_components=2, random_state=self._a.random_state)
        proj = pca.fit_transform(X)

        proj_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
        display_kde = KernelDensity(kernel="gaussian", bandwidth=DensityEstimate._scott_bandwidth(proj))
        display_kde.fit(proj)

        pad = 0.05
        x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
        y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
        x_pad, y_pad = (x_max - x_min) * pad, (y_max - y_min) * pad
        xx, yy = np.meshgrid(
            np.linspace(x_min - x_pad, x_max + x_pad, resolution),
            np.linspace(y_min - y_pad, y_max + y_pad, resolution),
        )
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        density = display_kde.score_samples(grid_pts).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(xx, yy, density, levels=15, cmap="Greys")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("Log-density (PCA-space, display only)", color="black")

        pts = proj_df
        if len(pts) > max_points:
            pts = pts.sample(n=max_points, random_state=0)
        ax.scatter(
            pts["PC1"], pts["PC2"],
            s=6, color="white", edgecolor="black", linewidth=0.3, alpha=0.6,
        )

        explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} var)")
        _title(ax, f"Density — PCA Projection ({len(self._a.columns)}D \u2192 2D)")

        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax

    def low_density_regions_table(self, top_n: int = 15, figsize: tuple[float, float] = (9, 6)):
        """Horizontal bar chart of the largest low-density "holes" by
        member count, mirroring ExactDuplicates.plot.group_sizes().
        Bars are hatched (matches the "bad outcome" convention used for
        cross_split/label_conflict) since these are gaps, not clusters."""
        table = self._a.low_density_regions()
        if table.empty:
            raise ValueError("No low-density regions found at the auto threshold.")

        top = table.head(top_n).iloc[::-1]
        labels = [f"hole {rid}" for rid in top["region_id"]]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(
            labels, top["n_points"],
            color="white", edgecolor="black", linewidth=1.2,
        )
        for b in bars:
            b.set_hatch("///")

        ax.set_xlabel("Rows in region")
        _title(ax, "Largest Low-Density Regions")

        ax.grid(axis="x", alpha=0.2)
        _style_axes(fig, ax)
        fig.tight_layout()
        return fig, ax