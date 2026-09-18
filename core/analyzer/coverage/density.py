"""
coverage.density — DensityEstimate analyzer.

Foundational analyzer for the Coverage Analysis Subsystem. Builds a
multidimensional density model of the training data (KDE or GMM),
exposes density scoring for arbitrary points, and surfaces low-density
("holes") and high-density (clusters) regions.

Everything else in the coverage subsystem (GapDetector,
ExtrapolationMapper, SufficiencyScorer) is built on top of this — it
plays the same foundational role that `profile.base` plays for
ExactDuplicates.

Usage:
    de = DensityEstimate(df, columns=["age", "income", "tenure"])
    de.density_summary()
    de.low_density_regions()
    de.high_density_clusters()
    de.row_density()

    de.plot.density_1d("income")
    de.plot.density_2d(("age", "income"))
    de.plot.density_projection(method="pca")
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from core.analyzer import Analyzer
from core.analyzer.analyzer import AnalyzerType
from core.visualizer.density import DensityEstimateVisualizer


# ----------------------------------------------------------------------
# Assumed shared helpers / base classes (match ExactDuplicates example).
# In the real codebase these are imported from the shared module; kept
# here as lightweight stand-ins so this file is self-contained and
# runnable for review/testing.
# ----------------------------------------------------------------------


def _validate_column_exists(df: pd.DataFrame, col: str, name: str) -> None:
        if col not in df.columns:
            raise ValueError(f"{name}={col!r} is not a column in the given dataframe.")




# ----------------------------------------------------------------------
# DensityEstimate
# ----------------------------------------------------------------------

DensityMethod = Literal["kde", "gmm"]


class DensityEstimate(Analyzer):
    """Multidimensional density model of the training data.

    Two estimator backends are supported via `method`, chosen as a
    strategy parameter rather than a subclass hierarchy — downstream
    analyzers (GapDetector, ExtrapolationMapper, SufficiencyScorer)
    only ever need `.score()` / `.density_grid()` / row-level density,
    and shouldn't have to special-case which estimator produced them.

        method="kde"  Kernel Density Estimation (sklearn.neighbors.KernelDensity).
                       Non-parametric, good default for small-to-medium
                       feature counts and non-Gaussian shapes. Bandwidth
                       matters a lot; if not supplied, Scott's rule is
                       used as a data-driven default.

        method="gmm"  Gaussian Mixture Model (sklearn.mixture.GaussianMixture).
                       Parametric, scales better to more dimensions/rows,
                       gives a smoother density surface, and its
                       component means/covariances double as a cheap
                       description of high-density clusters.

    Numeric columns only. Categorical/text columns should be encoded
    upstream (or excluded via `columns=`) before construction — density
    estimation over unordered categories isn't meaningful with either
    backend here.

    All features are z-scored internally (StandardScaler) before
    fitting, so that no single high-magnitude feature dominates
    distance/bandwidth calculations. Distances/densities reported back
    in low_density_regions()/high_density_clusters() are converted back
    to the original feature scale for interpretability.

    Usage:
        de = DensityEstimate(df, columns=["age", "income", "tenure"])
        de.density_summary()
        de.low_density_regions()
        de.high_density_clusters()
        de.row_density()
        de.plot.density_2d(("age", "income"))
    """

    id = "coverage.density"

    capability = "coverage.density"

    requires = {"profile.base"}

    provides = {capability}

    analyzer_type = AnalyzerType.DATASET

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        method: DensityMethod = "kde",
        bandwidth: Optional[float] = None,
        kernel: str = "gaussian",
        n_components: Optional[int] = None,
        max_components: int = 8,
        random_state: int = 0,
        sample_for_fit: Optional[int] = 50_000,
    ) -> None:
        super().__init__()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not columns:
                raise ValueError(
                    "No numeric columns found. Pass `columns=` explicitly "
                    "(encode categoricals upstream first)."
                )
        else:
            for c in columns:
                _validate_column_exists(df, c, "columns")
            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
            if non_numeric:
                raise ValueError(
                    f"DensityEstimate requires numeric columns; {non_numeric} are "
                    f"not numeric. Encode them upstream or drop from `columns=`."
                )

        if method not in ("kde", "gmm"):
            raise ValueError(f"method must be 'kde' or 'gmm', got {method!r}")

        if len(columns) < 1:
            raise ValueError("Need at least 1 column to estimate density.")

        n_rows_with_na = df[columns].isna().any(axis=1).sum()
        if n_rows_with_na > 0:
            raise ValueError(
                f"{n_rows_with_na} rows contain NaNs in the selected columns. "
                f"DensityEstimate does not impute — clean/impute upstream "
                f"before constructing this analyzer."
            )

        self.df = df
        self.columns = columns
        self.method = method
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_components = n_components
        self.max_components = max_components
        self.random_state = random_state
        self.sample_for_fit = sample_for_fit

        self._plot: Optional["DensityEstimateVisualizer"] = None

        self._scaler: Optional[StandardScaler] = None
        self._estimator = None
        self._fitted = False

        # caches
        self._row_log_density: Optional[np.ndarray] = None
        self._low_density_cache: Optional[pd.DataFrame] = None
        self._high_density_cache: Optional[pd.DataFrame] = None

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        self._fit()
        return super().analyze(ctx)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_data(self) -> pd.DataFrame:
        """Data used to fit the estimator, optionally subsampled for
        speed on large datasets. KDE in particular is O(n) per query
        and gets slow to fit/score above ~50k rows; GMM scales better
        but subsampling still speeds up fitting with negligible loss
        of density accuracy for typical dataset sizes."""
        data = self.df[self.columns]
        if self.sample_for_fit is not None and len(data) > self.sample_for_fit:
            return data.sample(n=self.sample_for_fit, random_state=self.random_state)
        return data

    def _fit(self) -> None:
        if self._fitted:
            return

        fit_df = self._fit_data()
        self._scaler = StandardScaler()
        X_fit = self._scaler.fit_transform(fit_df.values)

        if self.method == "kde":
            bw = self.bandwidth if self.bandwidth is not None else self._scott_bandwidth(X_fit)
            self._estimator = KernelDensity(kernel=self.kernel, bandwidth=bw)
            self._estimator.fit(X_fit)
        else:  # gmm
            n_components = self.n_components
            if n_components is None:
                n_components = self._select_gmm_components(X_fit)
            self._estimator = GaussianMixture(
                n_components=n_components,
                random_state=self.random_state,
                covariance_type="full",
            )
            self._estimator.fit(X_fit)

        self._fitted = True

    @staticmethod
    def _scott_bandwidth(X: np.ndarray) -> float:
        """Scott's rule of thumb: n^(-1/(d+4)), applied on standardized
        data. A reasonable data-driven default when the user hasn't
        tuned bandwidth by hand."""
        n, d = X.shape
        return float(n ** (-1.0 / (d + 4)))

    def _select_gmm_components(self, X: np.ndarray) -> int:
        """Pick number of GMM components via BIC over a small grid,
        capped at max_components (and at n_samples) to keep fitting
        cheap by default. Users who want a specific number of
        components should pass n_components= explicitly."""
        n = X.shape[0]
        upper = max(1, min(self.max_components, n - 1))
        candidates = list(range(1, upper + 1))
        best_bic = np.inf
        best_k = 1
        for k in candidates:
            gmm = GaussianMixture(
                n_components=k, random_state=self.random_state, covariance_type="full"
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        return best_k

    # ------------------------------------------------------------------
    # Core scoring primitives
    # ------------------------------------------------------------------

    def score(self, points: pd.DataFrame) -> np.ndarray:
        """Log-density at each row of `points` (must contain
        self.columns). This is the core primitive every downstream
        coverage analyzer (gaps, extrapolation, sufficiency) calls."""
        self._fit()
        for c in self.columns:
            _validate_column_exists(points, c, "points")

        X = self._scaler.transform(points[self.columns].values)

        if self.method == "kde":
            return self._estimator.score_samples(X)
        else:
            return self._estimator.score_samples(X)

    def row_density(self) -> pd.Series:
        """Log-density of every row in the original training df,
        cached. The per-row substrate that low_density_regions() /
        high_density_clusters() and downstream analyzers build on."""
        if self._row_log_density is None:
            self._row_log_density = self.score(self.df[self.columns])
        return pd.Series(self._row_log_density, index=self.df.index, name="log_density")

    def density_grid(
        self,
        dims: tuple[str, str],
        resolution: int = 100,
        padding: float = 0.05,
    ) -> dict:
        """Evaluate density over a 2D grid across `dims`, holding all
        other selected columns fixed at their training-data mean (in
        standardized space, i.e. 0). Used for 2D density heatmaps/
        contours — the primary visualization primitive for this class
        and for GapDetector's train-vs-reference comparison plots.

        Returns {"xx": grid, "yy": grid, "density": grid,
                 "x_edges": (min,max), "y_edges": (min,max)}.
        """
        self._fit()
        for d in dims:
            if d not in self.columns:
                raise ValueError(
                    f"{d!r} is not one of the columns this DensityEstimate was "
                    f"fit on ({self.columns})."
                )

        x_col, y_col = dims
        x_vals = self.df[x_col].values
        y_vals = self.df[y_col].values

        x_pad = (x_vals.max() - x_vals.min()) * padding
        y_pad = (y_vals.max() - y_vals.min()) * padding
        x_min, x_max = x_vals.min() - x_pad, x_vals.max() + x_pad
        y_min, y_max = y_vals.min() - y_pad, y_vals.max() + y_pad

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )

        # Build query points: grid over (x_col, y_col), other columns
        # fixed at their training mean.
        other_cols = [c for c in self.columns if c not in dims]
        n_grid_pts = xx.size
        query = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
        for c in other_cols:
            query[c] = self.df[c].mean()
        query = query[self.columns]  # column order must match fit order

        density = self.score(query).reshape(xx.shape)

        return {
            "xx": xx,
            "yy": yy,
            "density": density,
            "x_edges": (x_min, x_max),
            "y_edges": (y_min, y_max),
        }

    # ------------------------------------------------------------------
    # Region detection
    # ------------------------------------------------------------------

    def density_summary(self) -> dict:
        """Cheap top-line rollup: distribution of row-level log-density,
        and what fraction of training rows sit below an "auto" low-
        density threshold. Same role as duplicate_rate() — the first,
        cheapest thing to look at."""
        d = self.row_density()
        threshold = self._auto_threshold(d.values)
        return {
            "method": self.method,
            "n_rows": int(len(d)),
            "n_dims": len(self.columns),
            "log_density_min": float(d.min()),
            "log_density_max": float(d.max()),
            "log_density_mean": float(d.mean()),
            "log_density_median": float(d.median()),
            "low_density_threshold": float(threshold),
            "pct_rows_low_density": float((d.values < threshold).mean()),
        }

    @staticmethod
    def _auto_threshold(log_density: np.ndarray, quantile: float = 0.10) -> float:
        """Default low-density cutoff: 10th percentile of the observed
        row log-densities. Simple, robust, and matches the intuition
        that "low density" is relative to this dataset's own spread
        rather than an absolute value (log-density scale isn't
        comparable across different dimensionalities/bandwidths)."""
        return float(np.quantile(log_density, quantile))

    def low_density_regions(
        self,
        threshold: Union[float, Literal["auto"]] = "auto",
        eps: Optional[float] = None,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        """Cluster the low-density training rows (density below
        `threshold`) into spatially coherent "holes" using DBSCAN, so
        that scattered low-density points form describable regions
        rather than a flat list of individual rows.

        Each output row is one hole: centroid (in original feature
        scale), approximate radius, member count, and mean density.
        Points that don't fall into a DBSCAN cluster (noise, i.e.
        isolated low-density rows with no nearby low-density
        neighbors) are reported separately via `n_isolated_points` in
        density_summary()-style usage, not as their own hole rows,
        since a "region" implies more than one point.
        """
        if self._low_density_cache is not None and threshold == "auto":
            return self._low_density_cache

        d = self.row_density()
        thresh_val = self._auto_threshold(d.values) if threshold == "auto" else threshold
        low_mask = d.values < thresh_val

        if low_mask.sum() == 0:
            out = pd.DataFrame(
                columns=["region_id", "n_points", "mean_log_density", "centroid", "radius"]
            )
            if threshold == "auto":
                self._low_density_cache = out
            return out

        low_df = self.df.loc[low_mask, self.columns]
        X_low = self._scaler.transform(low_df.values)

        if eps is None:
            # Heuristic: eps as a fraction of the average pairwise
            # spread in standardized space, scaled down as dimensionality
            # grows (distances concentrate in high-D).
            eps = 0.5 * np.sqrt(len(self.columns))

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_low)

        rows = []
        for region_id in sorted(set(labels)):
            if region_id == -1:
                continue  # noise / isolated points, not a "region"
            member_mask = labels == region_id
            members = low_df.loc[member_mask]
            members_std = X_low[member_mask]
            centroid_std = members_std.mean(axis=0, keepdims=True)
            centroid_orig = self._scaler.inverse_transform(centroid_std)[0]
            radius = float(np.linalg.norm(members_std - centroid_std, axis=1).max())
            rows.append(
                {
                    "region_id": int(region_id),
                    "n_points": int(member_mask.sum()),
                    "mean_log_density": float(d.values[low_mask][member_mask].mean()),
                    "centroid": dict(zip(self.columns, centroid_orig.tolist())),
                    "radius": radius,
                }
            )

        out = pd.DataFrame(rows).sort_values("n_points", ascending=False).reset_index(drop=True) \
            if rows else pd.DataFrame(
                columns=["region_id", "n_points", "mean_log_density", "centroid", "radius"]
            )

        if threshold == "auto":
            self._low_density_cache = out
        return out

    def high_density_clusters(
        self,
        threshold: Union[float, Literal["auto"]] = "auto",
        eps: Optional[float] = None,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        """Mirror of low_density_regions(): clusters rows with density
        ABOVE `threshold` (default: 90th percentile) into coherent
        high-density clusters. Same output schema."""
        if self._high_density_cache is not None and threshold == "auto":
            return self._high_density_cache

        d = self.row_density()
        thresh_val = (
            float(np.quantile(d.values, 0.90)) if threshold == "auto" else threshold
        )
        high_mask = d.values > thresh_val

        if high_mask.sum() == 0:
            out = pd.DataFrame(
                columns=["region_id", "n_points", "mean_log_density", "centroid", "radius"]
            )
            if threshold == "auto":
                self._high_density_cache = out
            return out

        high_df = self.df.loc[high_mask, self.columns]
        X_high = self._scaler.transform(high_df.values)

        if eps is None:
            eps = 0.5 * np.sqrt(len(self.columns))

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_high)

        rows = []
        for region_id in sorted(set(labels)):
            if region_id == -1:
                continue
            member_mask = labels == region_id
            members_std = X_high[member_mask]
            centroid_std = members_std.mean(axis=0, keepdims=True)
            centroid_orig = self._scaler.inverse_transform(centroid_std)[0]
            radius = float(np.linalg.norm(members_std - centroid_std, axis=1).max())
            rows.append(
                {
                    "region_id": int(region_id),
                    "n_points": int(member_mask.sum()),
                    "mean_log_density": float(d.values[high_mask][member_mask].mean()),
                    "centroid": dict(zip(self.columns, centroid_orig.tolist())),
                    "radius": radius,
                }
            )

        out = pd.DataFrame(rows).sort_values("n_points", ascending=False).reset_index(drop=True) \
            if rows else pd.DataFrame(
                columns=["region_id", "n_points", "mean_log_density", "centroid", "radius"]
            )

        if threshold == "auto":
            self._high_density_cache = out
        return out

    @property
    def plot(self) -> "DensityEstimateVisualizer":
        if self._plot is None:
            self._plot = DensityEstimateVisualizer(self)
        return self._plot



