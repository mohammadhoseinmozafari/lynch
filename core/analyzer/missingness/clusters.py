from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from core.analyzer import Analyzer, AnalysisResult, AnalyzerType
from core.visualizer.missingness import MissingnessClustersVisualizer

Method = Literal["exact", "hierarchical"]

_TABLE_COLUMNS = ["cluster_id", "size", "rate", "n_missing_cols", "missing_columns"]


@dataclass
class ClusterResult:
    """Everything a visualizer or `rows_in` needs, computed once."""

    table: pd.DataFrame                 # one row per cluster, sorted by size desc
    labels: pd.Series                   # cluster_id per original row (index = df.index)
    varying_cols: list[str]
    pattern: pd.DataFrame               # cluster_id x varying_cols, fraction missing (0..1)
    linkage: Optional[np.ndarray] = None  # only for hierarchical
    method: str = "exact"
    meta: dict = field(default_factory=dict)


class MissingnessClusters(Analyzer):
    """Cluster rows by their missingness pattern.

    Usage:
        result = MissingnessClusters().clusters(df)                       # exact signatures
        result = MissingnessClusters().clusters(df, "hierarchical", k=6)  # coarser, wide data
        analyzer.rows_in(df, cluster_id=2)
    """

    id = "dataset.missingness.clusters"
    capability = "missingness.row_clusters"
    requires = {"profile.base"}
    provides = {capability}
    analyzer_type = AnalyzerType.DATASET

    # Above this many unique signatures, hierarchical clustering is run on
    # unique signatures (weighted) rather than raw rows.
    MAX_ROWS_FOR_PDIST = 5_000

    def __init__(self) -> None:
        super().__init__()
        self.visualizer = MissingnessClustersVisualizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clusters(
        self,
        df: pd.DataFrame,
        method: Method = "exact",
        k: int = 8,
        metric: str = "hamming",
    ) -> AnalysisResult:
        """Cluster rows by missingness pattern.

        method="exact": rows grouped by literal missingness signature.
            No hyperparameters; can yield many tiny clusters when
            missingness is scattered across many columns.
        method="hierarchical": agglomerative clustering on binary
            missingness vectors, cut into `k` clusters. Coarser and more
            stable on wide data; `missing_columns` is a >50%-of-cluster
            summary rather than exact.
        """
        result = self._compute(df, method=method, k=k, metric=metric)
        return AnalysisResult(
            raw=result,
            visualizer=self.visualizer.visualize_clusters,
        )

    def pattern_matrix(
        self, df: pd.DataFrame, method: Method = "exact", k: int = 8, metric: str = "hamming"
    ) -> AnalysisResult:
        result = self._compute(df, method=method, k=k, metric=metric)
        return AnalysisResult(raw=result, visualizer=self.visualizer.visualize_pattern)

    def dendrogram(self, df: pd.DataFrame, k: int = 8, metric: str = "hamming") -> AnalysisResult:
        result = self._compute(df, method="hierarchical", k=k, metric=metric)
        return AnalysisResult(raw=result, visualizer=self.visualizer.visualize_dendrogram)

    def rows_in(
        self,
        df: pd.DataFrame,
        cluster_id: int,
        method: Method = "exact",
        k: int = 8,
        metric: str = "hamming",
    ) -> pd.DataFrame:
        """Subset of `df` belonging to one cluster."""
        result = self._compute(df, method=method, k=k, metric=metric)
        if cluster_id not in set(result.table["cluster_id"]):
            raise ValueError(f"No cluster with id {cluster_id}")
        return df.loc[(result.labels == cluster_id).values]

    def summary(
        self, df: pd.DataFrame, method: Method = "exact", metric: str ="hamming", k: int = 8 
    ) -> dict:
        table = self._compute(df, method=method, k=k, metric=metric).table
        if table.empty:
            return {"n_clusters": 0, "largest_cluster_rate": 0.0, "complete_rows_rate": 0.0}
        return {
            "n_clusters": int(len(table)),
            "largest_cluster_rate": float(table["rate"].max()),
            "complete_rows_rate": float(table.loc[table["n_missing_cols"] == 0, "rate"].sum()),
        }

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, df: pd.DataFrame, method: Method, k: int, metric: str) -> ClusterResult:
        if method not in ("exact", "hierarchical"):
            raise ValueError(f"Invalid method: {method!r}. Expected 'exact' or 'hierarchical'.")

        mask = df.isna()
        # Only columns whose missingness varies carry signal; constant
        # columns would pad every signature identically.
        varying = list(mask.columns[mask.nunique() > 1])

        if not varying or len(df) == 0:
            return self._empty(df, varying, method)

        sub = mask[varying].to_numpy(dtype=bool)

        # Unique signatures + inverse mapping: O(n) via np.unique, no row-wise apply.
        sigs, inverse, counts = np.unique(
            sub, axis=0, return_inverse=True, return_counts=True
        )
        inverse = inverse.ravel()

        if method == "exact":
            sig_labels = np.arange(len(sigs))
            Z = None
        else:
            if len(sigs) < 2:
                raise ValueError("Need at least 2 distinct missingness patterns to cluster.")
            sig_labels, Z = self._hierarchical(sigs, counts, k, metric)

        row_labels = sig_labels[inverse]
        return self._build_result(
            df, varying, sigs, counts, sig_labels, row_labels, Z, method, k, metric
        )

    def _hierarchical(self, sigs, counts, k, metric):
        """Cluster unique signatures (not raw rows) — cheap and equivalent
        up to duplicate weighting, which we handle by expanding only when small."""
        data = sigs.astype(int)
        k = max(1, min(k, len(sigs)))
        distances = pdist(data, metric=metric)
        Z = linkage(distances, method="average")
        labels = fcluster(Z, t=k, criterion="maxclust") - 1
        return labels, Z

    def _build_result(
        self, df, varying, sigs, counts, sig_labels, row_labels, Z, method, k, metric
    ) -> ClusterResult:
        n = len(df)
        raw_ids = np.unique(row_labels)

        # size per raw cluster
        sizes = {cid: int(counts[sig_labels == cid].sum()) for cid in raw_ids}
        # renumber by size rank so cluster 0 is always the largest
        order = sorted(raw_ids, key=lambda c: -sizes[c])
        remap = {old: new for new, old in enumerate(order)}

        row_labels = np.vectorize(remap.get)(row_labels)
        sig_labels = np.vectorize(remap.get)(sig_labels)

        rows, pattern_rows = [], []
        for new_id in range(len(order)):
            in_cluster = sig_labels == new_id
            w = counts[in_cluster]
            # weighted fraction of rows missing each column
            col_rates = (sigs[in_cluster] * w[:, None]).sum(axis=0) / w.sum()
            missing_cols = [c for c, r in zip(varying, col_rates) if r > 0.5]
            size = int(w.sum())
            rows.append({
                "cluster_id": new_id,
                "size": size,
                "rate": size / n,
                "n_missing_cols": len(missing_cols),
                "missing_columns": missing_cols,
            })
            pattern_rows.append(col_rates)

        table = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
        pattern = pd.DataFrame(pattern_rows, columns=varying, index=table["cluster_id"])

        return ClusterResult(
            table=table,
            labels=pd.Series(row_labels, index=df.index, name="cluster_id"),
            varying_cols=varying,
            pattern=pattern,
            linkage=Z,
            method=method,
            meta={"k": k, "metric": metric, "n_rows": n},
        )

    def _empty(self, df, varying, method) -> ClusterResult:
        return ClusterResult(
            table=pd.DataFrame(columns=_TABLE_COLUMNS),
            labels=pd.Series(0, index=df.index, name="cluster_id", dtype=int),
            varying_cols=varying,
            pattern=pd.DataFrame(columns=varying),
            method=method,
            meta={"n_rows": len(df)},
        )