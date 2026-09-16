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

import pandas as pd
from .analyzer import Analyzer, AnalyzerType
from core.visualizer.duplicates import ExactDuplicatesVisualizer

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
    def plot(self) -> "ExactDuplicatesPlotter":
        if self._plot is None:
            self._plot = ExactDuplicatesPlotter(self)
        return self._plot