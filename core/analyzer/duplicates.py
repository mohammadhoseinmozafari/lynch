"""
ExactDuplicates: baseline duplicate check — two rows identical across
every column. Cheap, fully deterministic. Per spec (section 2), this is
the first check an expert runs because its presence is almost always a
pipeline bug, and severity is visible from the rate alone before any
further investigation.

Optional context:
- split_col: to distinguish within-split duplicates (recoverable — drop
  one) from cross-split duplicates (a validity problem — the model has
  already "seen" its test data).
- label_col: to distinguish duplicates with consistent labels (safe to
  drop one) from duplicates with conflicting labels (actively corrupt
  training signal; see LabelConsistency for the deeper analysis — this
  class only surfaces the raw conflict, not the Bayes-error implications).

Follows the same shape as the missingness analyzers: a pure-data Analyzer
class, plus a separate Plotter class exposed via a lazily-constructed
`.plot` property, so importing/instantiating the analyzer never touches
matplotlib.
"""

from __future__ import annotations

from typing import Mapping, Optional
import numpy as np
import pandas as pd
from .analyzer import Analyzer, AnalyzerType
from core.visualizer.duplicates import ExactDuplicatesVisualizer
from itertools import combinations
from typing import Literal, Mapping, Optional

from scipy.spatial.distance import pdist, squareform


def _validate_column_exists(df: pd.DataFrame, col: str, arg_name: str) -> None:
    if col not in df.columns:
        raise ValueError(f"{arg_name}={col!r} is not a column in df.")


class ExactDuplicates(Analyzer):
    """Exact row duplicate detection.

    Two ways to construct it, depending on when the split happened in
    your workflow:

    1. Split already applied, single combined dataframe with a column
       marking which split each row belongs to (split-after-check, or
       any case where you already have one df):

        ed = ExactDuplicates(df, split_col="split", label_col="target")

    2. Split already applied as SEPARATE dataframes with no shared split
       column (split-before-check — the common case when you did
       `train_df, test_df = train_test_split(...)` before ever running
       this analyzer):

        ed = ExactDuplicates.from_splits(
            {"train": train_df, "test": test_df},
            label_col="target",
        )

    Both paths end up with the same internal state (one combined df
    plus a split_col) — from_splits() just builds that combination for
    you instead of requiring you to pd.concat() it yourself, which is
    easy to forget and would silently hide cross-split duplicates if
    you ran the analyzer on train_df alone.

    Usage (either construction path):
        ed.duplicate_rate()
        ed.duplicate_groups()
        ed.cross_split_overlap()   # requires split_col (set by either path)
        ed.label_conflicts()       # requires label_col
        ed.plot.rate_by_split()
        ed.plot.label_conflict_breakdown()
    """

    id = "dataset.duplicates.exact"

    capability = "duplicates.exact"

    requires = {"profile.base"}

    provides = {capability}

    analyzer_type = AnalyzerType.DATASET

    def __init__(
        self,
        df: pd.DataFrame,
        split_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> None:
        super().__init__()

        if split_col is not None:
            _validate_column_exists(df, split_col, "split_col")
        if label_col is not None:
            _validate_column_exists(df, label_col, "label_col")

        self.df = df
        self.split_col = split_col
        self.label_col = label_col
        self._plot: Optional["ExactDuplicatesPlotter"] = None

        # Columns used to determine row identity. When split/label columns
        # are supplied they are excluded from the identity check itself —
        # two rows are "the same duplicate" based on their features, and
        # split/label are then read off as *properties* of that duplicate
        # group, not part of what defines it. This matches spec intent:
        # a row in train and its exact copy in test are the same
        # duplicate group straddling two splits, not two different
        # duplicate groups that happen to share values.
        exclude = {c for c in (split_col, label_col) if c is not None}
        self.identity_cols = [c for c in df.columns if c not in exclude]

        # Group id per row: NaN-safe row hash over identity_cols only.
        # fillna with a sentinel so NaN == NaN for grouping purposes
        # (pandas groupby already treats NaNs as equal within a group as
        # of modern versions, but being explicit keeps this robust).
        self._row_key = df[self.identity_cols].apply(
            lambda row: tuple(row.values), axis=1
        )

    @classmethod
    def from_splits(
        cls,
        splits: Mapping[str, pd.DataFrame],
        label_col: Optional[str] = None,
        split_col_name: str = "split",
    ) -> "ExactDuplicates":
        """Build an ExactDuplicates from separate per-split dataframes
        (e.g. train_df, test_df you already have as distinct objects,
        with no shared "split" column between them) rather than one
        combined dataframe.

        splits: mapping of split name -> dataframe, e.g.
            {"train": train_df, "test": test_df}
            or {"train": train_df, "val": val_df, "test": test_df}
        All dataframes must share the same columns.

        split_col_name: name to give the synthetic split column that
        gets attached internally. Must not already exist as a column
        in any of the supplied dataframes (raises otherwise) — if it
        does, pick a different split_col_name.

        Equivalent to concatenating the dataframes yourself and tagging
        each with its split, i.e.:
            train_df = train_df.assign(split="train")
            test_df = test_df.assign(split="test")
            combined = pd.concat([train_df, test_df], ignore_index=True)
            ExactDuplicates(combined, split_col="split")
        but removes the risk of forgetting that step (which would
        silently make cross-split leakage undetectable, since the
        analyzer would never see both splits at once).
        """
        if len(splits) < 2:
            raise ValueError(
                "from_splits() requires at least 2 splits to compare "
                "(got {}). For a single dataframe, use the regular "
                "constructor instead.".format(len(splits))
            )

        frames = []
        reference_cols = None
        for split_name, split_df in splits.items():
            if split_col_name in split_df.columns:
                raise ValueError(
                    f"split_col_name={split_col_name!r} already exists as a "
                    f"column in the {split_name!r} dataframe. Pass a "
                    f"different split_col_name to from_splits()."
                )
            if reference_cols is None:
                reference_cols = set(split_df.columns)
            elif set(split_df.columns) != reference_cols:
                raise ValueError(
                    f"All dataframes passed to from_splits() must share the "
                    f"same columns. {split_name!r} differs from the others: "
                    f"missing {reference_cols - set(split_df.columns)}, "
                    f"extra {set(split_df.columns) - reference_cols}."
                )
            frames.append(split_df.assign(**{split_col_name: split_name}))

        combined = pd.concat(frames, ignore_index=True)
        return cls(combined, split_col=split_col_name, label_col=label_col)

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        return super().analyze(ctx)

    # ------------------------------------------------------------------
    # Internal: duplicate groups (size > 1), computed once and cached
    # ------------------------------------------------------------------

    def _groups(self) -> pd.DataFrame:
        """One row per row of df, tagged with its group_id and group
        size. Cached because several public methods build on this."""
        if hasattr(self, "_groups_cache"):
            return self._groups_cache

        group_id = self._row_key.astype("category").cat.codes
        sizes = group_id.map(group_id.value_counts())

        out = pd.DataFrame(
            {
                "group_id": group_id.values,
                "group_size": sizes.values,
            },
            index=self.df.index,
        )
        self._groups_cache = out
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def duplicate_rate(self) -> dict:
        """Overall exact duplicate rate: how many rows are part of some
        duplicate group (size > 1), and what fraction of the dataset
        that represents. A row that is the "original" and a row that is
        its "copy" are both counted — this is a row-level rate, not a
        count of extra/removable rows (see duplicate_groups() for that)."""
        groups = self._groups()
        n = len(self.df)
        if n == 0:
            return {"n_rows": 0, "n_duplicate_rows": 0, "rate": 0.0, "n_groups": 0}

        is_dup = groups["group_size"] > 1
        return {
            "n_rows": int(n),
            "n_duplicate_rows": int(is_dup.sum()),
            "rate": float(is_dup.mean()),
            "n_groups": int(groups.loc[is_dup, "group_id"].nunique()),
        }

    def duplicate_groups(self) -> pd.DataFrame:
        """One row per duplicate group (size > 1): group_id, size,
        row indices, and the number of "removable" rows (size - 1) —
        the count you'd drop under a keep-one strategy."""
        groups = self._groups()
        dup = groups.loc[groups["group_size"] > 1]
        if dup.empty:
            return pd.DataFrame(
                columns=["group_id", "size", "removable_rows", "row_indices"]
            )

        out = (
            dup.groupby("group_id")
            .apply(lambda g: pd.Series({
                "size": int(g["group_size"].iloc[0]),
                "removable_rows": int(g["group_size"].iloc[0] - 1),
                "row_indices": list(g.index),
            }), include_groups=False)
            .reset_index()
            .sort_values("size", ascending=False)
            .reset_index(drop=True)
        )
        return out

    def rows_in(self, group_id: int) -> pd.DataFrame:
        """Return the subset of the original df in a given duplicate
        group, for inspection."""
        groups = self._groups()
        mask = groups["group_id"] == group_id
        if not mask.any():
            raise ValueError(f"No duplicate group with id {group_id}")
        return self.df.loc[mask]

    def cross_split_overlap(self) -> pd.DataFrame:
        """For each duplicate group, which split values it spans and
        whether it crosses split boundaries. Requires split_col.

        Cross-split duplicate groups are the validity-critical finding:
        the model has already seen its evaluation data.
        """
        if self.split_col is None:
            raise ValueError(
                "cross_split_overlap() requires two or more splits defined ")

        groups = self._groups()
        dup_groups = groups.loc[groups["group_size"] > 1]
        if dup_groups.empty:
            return pd.DataFrame(
                columns=["group_id", "size", "splits", "n_splits", "is_cross_split"]
            )

        split_values = self.df[self.split_col]
        merged = dup_groups.join(split_values.rename("split"))

        out = (
            merged.groupby("group_id")
            .agg(
                size=("group_size", "first"),
                splits=("split", lambda s: sorted(set(s.dropna().tolist()))),
            )
            .reset_index()
        )
        out["n_splits"] = out["splits"].apply(len)
        out["is_cross_split"] = out["n_splits"] > 1
        return out.sort_values(
            ["is_cross_split", "size"], ascending=[False, False]
        ).reset_index(drop=True)

    def label_conflicts(self) -> pd.DataFrame:
        """For each duplicate group, the distinct labels present and
        whether they conflict. Requires label_col.

        Duplicate rows with different labels are materially worse than
        duplicates with the same label — they actively contradict each
        other as training signal.
        """
        if self.label_col is None:
            raise ValueError(
                "label_conflicts() requires label_col to be set "
                "in the constructor."
            )

        groups = self._groups()
        dup_groups = groups.loc[groups["group_size"] > 1]
        if dup_groups.empty:
            return pd.DataFrame(
                columns=["group_id", "size", "labels", "n_distinct_labels", "has_conflict"]
            )

        label_values = self.df[self.label_col]
        merged = dup_groups.join(label_values.rename("label"))

        out = (
            merged.groupby("group_id")
            .agg(
                size=("group_size", "first"),
                labels=("label", lambda s: sorted(set(s.dropna().tolist()), key=str)),
            )
            .reset_index()
        )
        out["n_distinct_labels"] = out["labels"].apply(len)
        out["has_conflict"] = out["n_distinct_labels"] > 1
        return out.sort_values(
            ["has_conflict", "size"], ascending=[False, False]
        ).reset_index(drop=True)

    def summary(self) -> dict:
        """High-level rollup: overall rate, plus cross-split and label
        conflict rates if the relevant columns were supplied."""
        out = self.duplicate_rate()

        if self.split_col is not None:
            cross = self.cross_split_overlap()
            out["cross_split_groups"] = (
                int(cross["is_cross_split"].sum()) if not cross.empty else 0
            )
            out["cross_split_rate_of_groups"] = (
                float(cross["is_cross_split"].mean()) if not cross.empty else 0.0
            )

        if self.label_col is not None:
            conflicts = self.label_conflicts()
            out["conflicting_groups"] = (
                int(conflicts["has_conflict"].sum()) if not conflicts.empty else 0
            )
            out["conflict_rate_of_groups"] = (
                float(conflicts["has_conflict"].mean()) if not conflicts.empty else 0.0
            )

        return out

    @property
    def plot(self) -> "ExactDuplicatesVisualizer":
        if self._plot is None:
            self._plot = ExactDuplicatesVisualizer(self)
        return self._plot



"""
NearDuplicates: pairs of rows that are "almost" identical — differ in a
small number of columns, or differ only by small amounts in continuous
features. Per spec (section 4), near-duplicates are often more insidious
than exact duplicates because they pass naive uniqueness checks but
still corrupt evaluation the same way exact cross-split duplicates do.

Similarity is STRUCTURAL (column values), not semantic — "NYC" and
"New York City" are not detected as near-duplicates by this module.
That's a data-cleaning problem, not a duplicate-detection problem in
this diagnostic sense (explicitly out of scope per spec).

Similarity metric: a Gower-style blend of
- numeric columns: per-column min-max-scaled, similarity = 1 - scaled
  Euclidean distance across numeric columns (normalized by sqrt of
  column count). Euclidean rather than mean-absolute-difference is
  used deliberately: with few numeric columns (1-3, common in
  practice), an averaging metric lets closeness on just one column
  buy a large chunk of the similarity score, producing many
  false-positive "near-duplicates" among genuinely unrelated rows.
  Squaring the gaps (Euclidean) penalizes divergence on any single
  column more sharply, which cuts that false-positive rate a lot but
  does not eliminate it -- see the threshold guidance below.
- categorical columns: one-hot encoded, Jaccard similarity
blended by the relative count of numeric vs categorical columns, so
neither type dominates purely because of how many columns exist.

IMPORTANT — on threshold choice: with only 1-2 numeric columns, random
unrelated rows can still land at surprisingly high similarity purely
by chance (min-max scaling means "close" is relative to the full
column range, and real data often clusters near the middle of that
range). The default threshold is set high (0.97) specifically to
guard against this. Before trusting a chosen threshold, call
`false_positive_rate(threshold)` to see what fraction of a random
sample of unrelated pairs would clear that bar on your actual data --
if it's non-trivial, raise the threshold, add more columns to the
comparison, or narrow numeric_cols/categorical_cols to the columns
that actually define entity identity for your use case.

Two computation modes:
- method="exact": true O(n^2) pairwise comparison via scipy pdist.
  Fully faithful, but the module refuses to run it above
  `max_exact_rows` (default 5,000) and raises with a clear message
  instead of silently taking a very long time.
- method="lsh": approximate, for larger datasets. Rows are bucketed by
  a hash of a coarse "blocking key" (by default, the categorical
  columns' values, or a caller-supplied subset of columns) and only
  compared within-bucket. This is manual blocking, not true
  MinHash/LSH — documented here explicitly as an approximation that
  trades recall (near-duplicate pairs that land in different buckets
  are missed) for tractable runtime. If you need true LSH, a
  datasketch-based implementation is a reasonable future upgrade but
  is not implemented here to avoid adding that dependency.

Follows the same shape as the other analyzers: pure-data Analyzer
class, plotting delegated to a Plotter behind a lazily-constructed
`.plot` property.
"""



class NearDuplicates(Analyzer):
    """Near-duplicate (fuzzy) row detection.

    Usage:
        nd = NearDuplicates(df, numeric_cols=["amount", "age"],
                             categorical_cols=["region", "category"])
        nd.find_pairs(threshold=0.9)

        nd = NearDuplicates(df, numeric_cols=[...], categorical_cols=[...],
                             split_col="split", label_col="target")
        nd.pairs_by_scope()
        nd.differing_columns_summary()

        # large dataset -> approximate mode
        nd = NearDuplicates(df, numeric_cols=[...], categorical_cols=[...],
                             method="lsh", block_cols=["region"])
        nd.find_pairs(threshold=0.9)
    """

    id = "dataset.duplicates.near"

    capability = "duplicates.near"

    requires = {"profile.base"}

    provides = {capability}

    analyzer_type = AnalyzerType.DATASET

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[list[str]] = None,
        categorical_cols: Optional[list[str]] = None,
        split_col: Optional[str] = None,
        label_col: Optional[str] = None,
        method: Literal["exact", "lsh"] = "exact",
        threshold: float = 0.97,
        block_cols: Optional[list[str]] = None,
        max_exact_rows: int = 5000,
    ) -> None:
        super().__init__()

        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.0, 1.0].")

        for col in (split_col, label_col):
            if col is not None:
                _validate_column_exists(df, col, "split_col/label_col")

        exclude = {c for c in (split_col, label_col) if c is not None}

        if numeric_cols is None and categorical_cols is None:
            # auto-infer from dtype, excluding split/label columns
            candidate_cols = [c for c in df.columns if c not in exclude]
            numeric_cols = [
                c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])
            ]
            categorical_cols = [
                c for c in candidate_cols if c not in numeric_cols
            ]
        else:
            numeric_cols = numeric_cols or []
            categorical_cols = categorical_cols or []

        for col in numeric_cols:
            _validate_column_exists(df, col, "numeric_cols")
        for col in categorical_cols:
            _validate_column_exists(df, col, "categorical_cols")

        overlap = set(numeric_cols) & set(categorical_cols)
        if overlap:
            raise ValueError(
                f"Columns cannot be in both numeric_cols and categorical_cols: {overlap}"
            )
        if not numeric_cols and not categorical_cols:
            raise ValueError(
                "No numeric or categorical columns to compare. Pass "
                "numeric_cols/categorical_cols explicitly, or ensure df "
                "has comparable columns beyond split_col/label_col."
            )

        if method not in ("exact", "lsh"):
            raise ValueError(f"Invalid method: {method!r}. Expected 'exact' or 'lsh'.")

        if block_cols is not None:
            for col in block_cols:
                _validate_column_exists(df, col, "block_cols")

        self.df = df
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.split_col = split_col
        self.label_col = label_col
        self.method = method
        self.threshold = threshold
        self.block_cols = block_cols or list(categorical_cols)
        self.max_exact_rows = max_exact_rows
        self._plot: Optional["NearDuplicatesPlotter"] = None

        if method == "exact" and len(df) > max_exact_rows:
            raise ValueError(
                f"method='exact' requested on {len(df):,} rows, which exceeds "
                f"max_exact_rows={max_exact_rows:,}. Exact pairwise comparison "
                f"is O(n^2) and will not scale here. Either pass "
                f"method='lsh' (approximate, blocked comparison), raise "
                f"max_exact_rows explicitly if you understand the cost, or "
                f"work on a sample."
            )

    def analyze(self, ctx: AnalysisContext) -> ProfileNamespace:
        return super().analyze(ctx)

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _similarity_matrix_exact(self, indices: pd.Index) -> np.ndarray:
        """Condensed pairwise similarity (scipy pdist format) for the
        given row indices, blending numeric + categorical similarity."""
        sub = self.df.loc[indices]
        n = len(sub)
        n_num = len(self.numeric_cols)
        n_cat = len(self.categorical_cols)

        if n_num > 0:
            num = sub[self.numeric_cols].astype(float).values
            col_min = np.nanmin(num, axis=0)
            col_max = np.nanmax(num, axis=0)
            col_range = col_max - col_min
            col_range[col_range == 0] = 1.0
            num_scaled = (num - col_min) / col_range
            # pairwise scaled Euclidean distance, normalized by sqrt(n_num)
            # so it lands roughly in [0, 1] regardless of column count,
            # then converted to similarity. Euclidean (not cityblock) is
            # used deliberately: squaring the per-column gaps penalizes
            # a single large divergence more sharply, which matters a lot
            # when there are only 1-2 numeric columns -- with cityblock
            # (mean absolute difference), being close on one of two
            # columns already buys half the similarity score, which
            # produces a high false-positive rate against unrelated
            # random rows. See NearDuplicates docstring for the default
            # threshold guidance this implies.
            num_sim = 1 - pdist(num_scaled, metric="euclidean") / np.sqrt(n_num)
        else:
            num_sim = None

        if n_cat > 0:
            cat_onehot = pd.get_dummies(
                sub[self.categorical_cols].astype(str)
            ).values.astype(bool)
            cat_sim = 1 - pdist(cat_onehot, metric="jaccard")
        else:
            cat_sim = None

        if num_sim is not None and cat_sim is not None:
            w_num = n_num / (n_num + n_cat)
            w_cat = n_cat / (n_num + n_cat)
            blended = w_num * num_sim + w_cat * cat_sim
        elif num_sim is not None:
            blended = num_sim
        else:
            blended = cat_sim

        return np.nan_to_num(blended, nan=0.0)

    def _blocks(self) -> dict:
        """Bucket row indices by a hash of block_cols. Rows in different
        buckets are never compared under method='lsh' -- this is the
        approximation. If block_cols is empty (no categorical columns
        and none supplied), falls back to a single block, which is
        equivalent to exact mode."""
        if not self.block_cols:
            return {"__all__": self.df.index}

        keys = self.df[self.block_cols].astype(str).apply(
            lambda row: tuple(row.values), axis=1
        )
        buckets: dict = {}
        for idx, key in keys.items():
            buckets.setdefault(key, []).append(idx)
        return {k: pd.Index(v) for k, v in buckets.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_pairs(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """All row pairs with similarity >= threshold. Columns:
        row_i, row_j, similarity, differing_columns (list of column
        names where the two rows' values differ -- categorical: not
        equal; numeric: not exactly equal. Useful for
        differing_columns_summary() and for spotting label-only diffs).

        method='exact': true pairwise comparison across all rows.
        method='lsh': only compares rows within the same block (see
        _blocks()); pairs spanning two blocks are never considered,
        by design -- this is the recall/runtime tradeoff.
        """
        th = threshold if threshold is not None else self.threshold
        if not 0.0 < th <= 1.0:
            raise ValueError("threshold must be in (0.0, 1.0].")

        results = []

        if self.method == "exact":
            blocks = {"__all__": self.df.index}
        else:
            blocks = self._blocks()

        compare_cols = self.numeric_cols + self.categorical_cols

        for _, idx in blocks.items():
            if len(idx) < 2:
                continue
            sims = self._similarity_matrix_exact(idx)
            sim_matrix = squareform(sims)
            idx_list = list(idx)
            n = len(idx_list)
            for i, j in combinations(range(n), 2):
                sim = sim_matrix[i, j]
                if sim >= th:
                    row_i, row_j = idx_list[i], idx_list[j]
                    differing = [
                        c for c in compare_cols
                        if not _values_equal(self.df.at[row_i, c], self.df.at[row_j, c])
                    ]
                    results.append({
                        "row_i": row_i,
                        "row_j": row_j,
                        "similarity": round(float(sim), 4),
                        "differing_columns": differing,
                        "n_differing": len(differing),
                    })

        if not results:
            return pd.DataFrame(
                columns=["row_i", "row_j", "similarity", "differing_columns", "n_differing"]
            )

        return (
            pd.DataFrame(results)
            .sort_values("similarity", ascending=False)
            .reset_index(drop=True)
        )

    def pairs_by_scope(self, threshold: Optional[float] = None) -> dict:
        """Split find_pairs() results into within-split vs cross-split
        counts/rates. Requires split_col. Cross-split near-duplicates
        carry the same evaluation-validity risk as cross-split exact
        duplicates."""
        if self.split_col is None:
            raise ValueError(
                "pairs_by_scope() requires split_col to be set in the constructor."
            )

        pairs = self.find_pairs(threshold=threshold)
        if pairs.empty:
            return {"within_split": 0, "cross_split": 0, "cross_split_rate": 0.0}

        split_values = self.df[self.split_col]
        is_cross = [
            split_values.at[r["row_i"]] != split_values.at[r["row_j"]]
            for _, r in pairs.iterrows()
        ]
        pairs = pairs.assign(is_cross_split=is_cross)

        within = int((~pairs["is_cross_split"]).sum())
        cross = int(pairs["is_cross_split"].sum())
        total = within + cross
        return {
            "within_split": within,
            "cross_split": cross,
            "cross_split_rate": float(cross / total) if total else 0.0,
            "pairs": pairs,
        }

    def differing_columns_summary(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """For all near-duplicate pairs, how often each column is the
        one that differs. A column that is disproportionately often the
        ONLY differing column is a mislabeling signal if that column is
        label_col (per spec: a near-duplicate pair differing only in
        the label looks like a labeling error wearing a near-duplicate
        costume, not a genuine structural near-duplicate)."""
        pairs = self.find_pairs(threshold=threshold)
        if pairs.empty:
            return pd.DataFrame(columns=["column", "times_differing", "times_sole_diff"])

        compare_cols = self.numeric_cols + self.categorical_cols
        counts = {c: 0 for c in compare_cols}
        sole_diff_counts = {c: 0 for c in compare_cols}

        for cols in pairs["differing_columns"]:
            for c in cols:
                counts[c] += 1
            if len(cols) == 1:
                sole_diff_counts[cols[0]] += 1

        out = pd.DataFrame({
            "column": list(counts.keys()),
            "times_differing": list(counts.values()),
            "times_sole_diff": [sole_diff_counts[c] for c in counts.keys()],
        })

        if self.label_col is not None and self.label_col not in out["column"].values:
            # label_col is tracked separately from compare_cols by design
            # (it's not part of the similarity computation) -- but the
            # spec explicitly wants to know if label is the "only" diff,
            # so check it directly here.
            label_only = 0
            label_values = self.df[self.label_col]
            for _, r in pairs.iterrows():
                if len(r["differing_columns"]) == 0:
                    # identical on all compared columns; check label separately
                    if label_values.at[r["row_i"]] != label_values.at[r["row_j"]]:
                        label_only += 1
            out = pd.concat([out, pd.DataFrame([{
                "column": f"{self.label_col} (label, not in similarity calc)",
                "times_differing": label_only,
                "times_sole_diff": label_only,
            }])], ignore_index=True)

        return out.sort_values("times_sole_diff", ascending=False).reset_index(drop=True)

    def near_vs_exact_ratio(self, exact_duplicates, threshold: Optional[float] = None) -> dict:
        """Ratio of near-duplicate pair count to exact-duplicate pair
        count. Takes an already-constructed ExactDuplicates instance
        (composition, not recomputation) so the two counts are
        comparable and consistent.

        A high ratio (many near-dupes, few exact) suggests augmentation
        artifacts or measurement noise rather than a copy-paste bug.
        """
        near_pairs = self.find_pairs(threshold=threshold)
        n_near = len(near_pairs)

        exact_groups = exact_duplicates.duplicate_groups()
        # convert exact duplicate groups into an equivalent pair count
        # (a group of size k contributes C(k,2) pairs) for a fair
        # apples-to-apples comparison against near_pairs, which is
        # already pair-shaped.
        if exact_groups.empty:
            n_exact_pairs = 0
        else:
            n_exact_pairs = int(
                exact_groups["size"].apply(lambda k: k * (k - 1) // 2).sum()
            )

        return {
            "n_near_pairs": n_near,
            "n_exact_pairs": n_exact_pairs,
            "ratio": float(n_near / n_exact_pairs) if n_exact_pairs else float("inf") if n_near else 0.0,
        }

    def false_positive_rate(
        self,
        threshold: Optional[float] = None,
        n_samples: int = 2000,
        seed: int = 0,
    ) -> dict:
        """Empirically estimate how often UNRELATED rows would clear
        the similarity threshold by chance, on this actual dataset.
        Shuffles each column independently (destroying any real
        relationship between rows while preserving each column's
        marginal distribution), then measures what fraction of random
        pairs from the shuffled data still score >= threshold.

        A high rate here means the threshold is too lenient for this
        column set -- e.g. because there are very few numeric columns,
        or they're highly concentrated/low-variance -- and found
        near-duplicate pairs should be treated skeptically until the
        threshold is raised or more columns are included.
        """
        th = threshold if threshold is not None else self.threshold
        rng = np.random.default_rng(seed)

        shuffled = self.df[self.numeric_cols + self.categorical_cols].copy()
        for c in shuffled.columns:
            shuffled[c] = rng.permutation(shuffled[c].values)

        n = min(n_samples, len(shuffled))
        sample_idx = shuffled.sample(n, random_state=seed).index if n < len(shuffled) else shuffled.index

        # reuse the similarity computation by temporarily pointing at
        # the shuffled frame's values via a lightweight stand-in
        original_df = self.df
        self.df = shuffled
        try:
            sims = self._similarity_matrix_exact(sample_idx)
        finally:
            self.df = original_df

        return {
            "threshold": th,
            "n_pairs_checked": len(sims),
            "false_positive_rate": float((sims >= th).mean()),
        }

    def summary(self, threshold: Optional[float] = None) -> dict:
        pairs = self.find_pairs(threshold=threshold)
        out = {
            "n_pairs": len(pairs),
            "threshold": threshold if threshold is not None else self.threshold,
            "method": self.method,
        }
        if not pairs.empty:
            out["mean_similarity"] = float(pairs["similarity"].mean())
            out["min_similarity"] = float(pairs["similarity"].min())

        if self.split_col is not None:
            scope = self.pairs_by_scope(threshold=threshold)
            out["within_split_pairs"] = scope["within_split"]
            out["cross_split_pairs"] = scope["cross_split"]
            out["cross_split_rate"] = scope["cross_split_rate"]

        return out

    @property
    def plot(self) -> "NearDuplicatesPlotter":
        if self._plot is None:
            self._plot = NearDuplicatesPlotter(self)
        return self._plot


def _values_equal(a, b) -> bool:
    """NaN-safe equality check used for differing_columns."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b