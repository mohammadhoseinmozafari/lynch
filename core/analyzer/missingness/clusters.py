from __future__ import annotations

from typing import   Literal, Optional
import pandas as pd
from core.visualizer.missingness import  MissingnessClustersVisualizer
from core.analyzer import Analyzer, AnalyzerType

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class MissingnessClusters(Analyzer):
    """Cluster rows by their missingness pattern.
 
    Usage:
        mcl = MissingnessClusters(df)
        mcl.clusters()                          # exact signature groups
        mcl.clusters(method="hierarchical", k=6) # coarser, for wide data
        mcl.rows_in(cluster_id=2)                # inspect one cluster
        mcl.plot.sizes()
        mcl.plot.pattern_matrix()
        mcl.plot.dendrogram()
    """
 
    id = "dataset.missingness.clusters"
 
    capability = "missingness.row_clusters"
 
    requires = {"profile.base"}
 
    provides = {capability}
 
    analyzer_type = AnalyzerType.DATASET
 
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.mask = df.isna()
        self._plot: Optional["MissingnessClustersVisualizer"] = None
 
        # only columns that actually vary in missingness carry signal;
        # all-missing / all-present columns would just pad every
        # signature identically and add noise to the distance metric.
        self.varying_cols = list(self.mask.columns[self.mask.nunique() > 1])
 
    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        return super().analyze(ctx)
 
    # ------------------------------------------------------------------
    # Exact-signature clustering
    # ------------------------------------------------------------------
 
    def _exact_clusters(self) -> pd.DataFrame:
        if not self.varying_cols:
            return pd.DataFrame(
                columns=["cluster_id", "size", "rate", "n_missing_cols", "missing_columns"]
            )
 
        sub = self.mask[self.varying_cols]
        # group rows by identical missingness signature
        signatures = sub.apply(tuple, axis=1)
        groups = signatures.value_counts()
 
        rows = []
        for i, (sig, size) in enumerate(groups.items()):
            missing_cols = [c for c, is_missing in zip(self.varying_cols, sig) if is_missing]
            rows.append({
                "cluster_id": i,
                "size": int(size),
                "rate": float(size / len(self.df)),
                "n_missing_cols": len(missing_cols),
                "missing_columns": missing_cols,
            })
 
        out = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
        out["cluster_id"] = range(len(out))  # renumber by size rank
        self._signatures = signatures  # cache for rows_in()
        self._cluster_map = {row["cluster_id"]: i for i, row in enumerate(rows)}
        return out
 
    # ------------------------------------------------------------------
    # Hierarchical clustering (coarser, for wide data)
    # ------------------------------------------------------------------
 
    def _hierarchical_clusters(self, k: int, metric: str = "hamming") -> pd.DataFrame:
        if not self.varying_cols:
            return pd.DataFrame(
                columns=["cluster_id", "size", "rate", "n_missing_cols", "missing_columns"]
            )
 
        sub = self.mask[self.varying_cols].astype(int)
        if len(sub) < 2:
            raise ValueError("Need at least 2 rows to cluster.")
 
        distances = pdist(sub.values, metric=metric)
        Z = linkage(distances, method="average")
        labels = fcluster(Z, t=k, criterion="maxclust")
 
        self._linkage = Z  # cache for dendrogram plot
        self._hier_labels = labels
 
        rows = []
        for cid in sorted(set(labels)):
            row_idx = labels == cid
            size = int(row_idx.sum())
            # columns missing in >50% of the cluster's rows describe it
            cluster_mask = sub.values[row_idx]
            col_rates = cluster_mask.mean(axis=0)
            missing_cols = [c for c, r in zip(self.varying_cols, col_rates) if r > 0.5]
            rows.append({
                "cluster_id": int(cid) - 1,
                "size": size,
                "rate": float(size / len(self.df)),
                "n_missing_cols": len(missing_cols),
                "missing_columns": missing_cols,
            })
 
        return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def clusters(
        self,
        method: Literal["exact", "hierarchical"] = "exact",
        k: int = 8,
        metric: str = "hamming",
    ) -> pd.DataFrame:
        """
        Return one row per missingness cluster: cluster_id, size, rate
        (fraction of all rows), n_missing_cols, and which columns define
        the pattern.
 
        method="exact": rows are grouped by their literal missingness
        signature. No hyperparameters. Can produce many small clusters
        if missingness is scattered/independent across many columns.
 
        method="hierarchical": rows are agglomeratively clustered on
        their binary missingness vectors and cut into `k` clusters.
        Coarser and more stable on wide data, but the reported
        "missing_columns" is a >50%-of-cluster summary, not exact.
        """
        if method == "exact":
            return self._exact_clusters()
        elif method == "hierarchical":
            return self._hierarchical_clusters(k=k, metric=metric)
        else:
            raise ValueError(
                f"Invalid method: {method!r}. Expected 'exact' or 'hierarchical'."
            )
 
    def rows_in(self, cluster_id: int, method: Literal["exact", "hierarchical"] = "exact",
                k: int = 8, metric: str = "hamming") -> pd.DataFrame:
        """Return the subset of the original df belonging to a given
        cluster, so you can inspect why the pattern occurs."""
        if method == "exact":
            if not hasattr(self, "_signatures"):
                self._exact_clusters()
            table = self._exact_clusters()
            if cluster_id not in table["cluster_id"].values:
                raise ValueError(f"No cluster with id {cluster_id}")
            target_sig = tuple(
                col in table.loc[table["cluster_id"] == cluster_id, "missing_columns"].iloc[0]
                for col in self.varying_cols
            )
            row_mask = self._signatures == target_sig
            return self.df.loc[row_mask]
        else:
            self._hierarchical_clusters(k=k, metric=metric)
            row_mask = self._hier_labels == (cluster_id + 1)
            return self.df.loc[row_mask]
 
    def summary(self, method: Literal["exact", "hierarchical"] = "exact", k: int = 8) -> dict:
        table = self.clusters(method=method, k=k)
        if table.empty:
            return {
                "n_clusters": 0,
                "largest_cluster_rate": 0.0,
                "complete_rows_rate": 0.0,
            }
        complete = table.loc[table["n_missing_cols"] == 0, "rate"].sum()
        return {
            "n_clusters": int(len(table)),
            "largest_cluster_rate": float(table["rate"].max()),
            "complete_rows_rate": float(complete),
        }
 
    @property
    def plot(self) -> "MissingnessClustersVisualizer":
        if self._plot is None:
            self._plot = MissingnessClustersVisualizer(self)
        return self._plot
 