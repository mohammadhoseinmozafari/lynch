"""
Test data generators for ExactDuplicates.

Each function returns dataframe(s) with KNOWN, PLANTED duplicate
patterns, plus the expected counts as a dict, so tests can assert
against ground truth rather than just "did it run without crashing."

Usage:
    df, expected = make_clean_dataset()
    df, expected = make_within_split_duplicates()
    df, expected = make_cross_split_leakage()
    df, expected = make_label_conflicts()
    train_df, val_df, test_df, expected = make_three_way_split()
    bad_a, bad_b = make_mismatched_schema_pair()
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _base_rows(n: int, seed: int) -> pd.DataFrame:
    """Unique, non-duplicated base rows to build scenarios on top of."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_id": rng.integers(10_000, 99_999, n),
        "amount": rng.normal(100, 25, n).round(2),
        "category": rng.choice(["a", "b", "c", "d"], n),
        "target": rng.integers(0, 2, n),
    })


# ----------------------------------------------------------------------
# 1. Clean dataset — no duplicates at all (negative control)
# ----------------------------------------------------------------------

def make_clean_dataset(n: int = 500, seed: int = 0):
    df = _base_rows(n, seed).reset_index(drop=True)
    # guarantee uniqueness on the identity columns in case of random collision
    df = df.drop_duplicates(subset=["user_id", "amount", "category"]).reset_index(drop=True)

    expected = {
        "n_rows": len(df),
        "n_duplicate_rows": 0,
        "n_groups": 0,
    }
    return df, expected


# ----------------------------------------------------------------------
# 2. Within-split duplicates only — no split/label columns at all
# ----------------------------------------------------------------------

def make_within_split_duplicates(n: int = 500, n_dup_pairs: int = 15, seed: int = 1):
    rng = np.random.default_rng(seed)
    df = _base_rows(n, seed)
    df = df.drop_duplicates(subset=["user_id", "amount", "category"]).reset_index(drop=True)

    dupe_rows = df.sample(n_dup_pairs, random_state=seed).copy()
    df = pd.concat([df, dupe_rows], ignore_index=True)

    expected = {
        "n_rows": len(df),
        "n_duplicate_rows": n_dup_pairs * 2,
        "n_groups": n_dup_pairs,
    }
    return df, expected


# ----------------------------------------------------------------------
# 3. Cross-split leakage — single combined df with a split column
# ----------------------------------------------------------------------

def make_cross_split_leakage(
    n_train: int = 600,
    n_test: int = 200,
    n_leaked: int = 10,
    seed: int = 2,
):
    rng = np.random.default_rng(seed)

    train_df = _base_rows(n_train, seed)
    train_df = train_df.drop_duplicates(subset=["user_id", "amount", "category"]).reset_index(drop=True)

    test_df = _base_rows(n_test, seed + 1)
    test_df = test_df.drop_duplicates(subset=["user_id", "amount", "category"]).reset_index(drop=True)

    # remove any accidental overlap between train/test before planting the real leak
    key_cols = ["user_id", "amount", "category"]
    train_keys = set(map(tuple, train_df[key_cols].values.tolist()))
    test_df = test_df[~test_df[key_cols].apply(tuple, axis=1).isin(train_keys)].reset_index(drop=True)

    # plant exact leakage: copy real train rows verbatim into test
    leaked = train_df.sample(n_leaked, random_state=seed).copy()

    train_tagged = train_df.assign(split="train")
    test_tagged = pd.concat([test_df, leaked], ignore_index=True).assign(split="test")

    combined = pd.concat([train_tagged, test_tagged], ignore_index=True)

    expected = {
        "n_rows": len(combined),
        "cross_split_groups": n_leaked,
        "n_groups": n_leaked,  # no within-split dupes planted here, so all groups are cross-split
    }
    return combined, expected


# ----------------------------------------------------------------------
# 4. Label conflicts — same features, contradictory labels
# ----------------------------------------------------------------------

def make_label_conflicts(n: int = 500, n_conflicts: int = 8, seed: int = 3):
    df = _base_rows(n, seed)
    df = df.drop_duplicates(subset=["user_id", "amount", "category"]).reset_index(drop=True)

    base = df.sample(n_conflicts, random_state=seed).copy()
    flipped = base.copy()
    flipped["target"] = 1 - flipped["target"]  # guaranteed opposite label

    df = pd.concat([df, base, flipped], ignore_index=True)

    expected = {
        "n_rows": len(df),
        "conflicting_groups": n_conflicts,
        "n_groups": n_conflicts,  # each conflict forms its own group of size 3 (1 original + base + flipped copy... )
    }
    return df, expected


# ----------------------------------------------------------------------
# 5. Three-way split (train/val/test) as SEPARATE dataframes,
#    for testing from_splits() with more than 2 splits
# ----------------------------------------------------------------------

def make_three_way_split(
    n_train: int = 500,
    n_val: int = 150,
    n_test: int = 150,
    n_train_val_leak: int = 5,
    n_train_test_leak: int = 7,
    seed: int = 4,
):
    train_df = _base_rows(n_train, seed).drop_duplicates(
        subset=["user_id", "amount", "category"]
    ).reset_index(drop=True)

    val_df = _base_rows(n_val, seed + 10).drop_duplicates(
        subset=["user_id", "amount", "category"]
    ).reset_index(drop=True)

    test_df = _base_rows(n_test, seed + 20).drop_duplicates(
        subset=["user_id", "amount", "category"]
    ).reset_index(drop=True)

    key_cols = ["user_id", "amount", "category"]
    train_keys = set(map(tuple, train_df[key_cols].values.tolist()))
    val_df = val_df[~val_df[key_cols].apply(tuple, axis=1).isin(train_keys)].reset_index(drop=True)
    test_df = test_df[~test_df[key_cols].apply(tuple, axis=1).isin(train_keys)].reset_index(drop=True)

    train_val_leak = train_df.sample(n_train_val_leak, random_state=seed).copy()
    train_test_leak = train_df.sample(n_train_test_leak, random_state=seed + 1).copy()

    val_df = pd.concat([val_df, train_val_leak], ignore_index=True)
    test_df = pd.concat([test_df, train_test_leak], ignore_index=True)

    expected = {
        "n_groups": n_train_val_leak + n_train_test_leak,
        "cross_split_groups": n_train_val_leak + n_train_test_leak,
        "train_val_leak": n_train_val_leak,
        "train_test_leak": n_train_test_leak,
    }
    return train_df, val_df, test_df, expected


# ----------------------------------------------------------------------
# 6. Mismatched schema pair — for testing from_splits() error handling
# ----------------------------------------------------------------------

def make_mismatched_schema_pair(seed: int = 5):
    train_df = _base_rows(200, seed)
    test_df = _base_rows(100, seed + 1).rename(columns={"amount": "amt"})  # column name mismatch
    return train_df, test_df


# ----------------------------------------------------------------------
# 7. Split-column-name collision — for testing from_splits() error handling
# ----------------------------------------------------------------------

def make_split_col_collision_pair(seed: int = 6):
    train_df = _base_rows(200, seed).assign(split="already_here")
    test_df = _base_rows(100, seed + 1)
    return train_df, test_df


# ----------------------------------------------------------------------
# 8. Everything at once — realistic "messy real dataset" combining
#    within-split dupes, cross-split leakage, and label conflicts
#    together, as separate train/test dataframes
# ----------------------------------------------------------------------

def make_realistic_mixed_scenario(
    n_train: int = 800,
    n_test: int = 200,
    n_within_train_dupes: int = 12,
    n_leaked_clean: int = 6,
    n_leaked_conflicting: int = 3,
    seed: int = 7,
):
    """
    n_leaked_clean: rows copied verbatim from train into test (label agrees)
    n_leaked_conflicting: rows copied from train into test but with the
        label flipped in the test copy (leakage AND label conflict at once)
    """
    train_df = _base_rows(n_train, seed).drop_duplicates(
        subset=["user_id", "amount", "category"]
    ).reset_index(drop=True)
    test_df = _base_rows(n_test, seed + 1).drop_duplicates(
        subset=["user_id", "amount", "category"]
    ).reset_index(drop=True)

    key_cols = ["user_id", "amount", "category"]
    train_keys = set(map(tuple, train_df[key_cols].values.tolist()))
    test_df = test_df[~test_df[key_cols].apply(tuple, axis=1).isin(train_keys)].reset_index(drop=True)

    # within-train duplicates
    within_dupes = train_df.sample(n_within_train_dupes, random_state=seed).copy()
    train_df = pd.concat([train_df, within_dupes], ignore_index=True)

    # clean leakage: train row copied into test, same label
    clean_leak = train_df.sample(n_leaked_clean, random_state=seed + 2).copy()

    # conflicting leakage: train row copied into test, label flipped
    conflict_leak = train_df.sample(n_leaked_conflicting, random_state=seed + 3).copy()
    conflict_leak["target"] = 1 - conflict_leak["target"]

    test_df = pd.concat([test_df, clean_leak, conflict_leak], ignore_index=True)

    expected = {
        "within_train_groups": n_within_train_dupes,
        "cross_split_groups": n_leaked_clean + n_leaked_conflicting,
        "conflicting_groups": n_leaked_conflicting,
    }
    return train_df, test_df, expected


if __name__ == "__main__":
    # quick self-check that every generator runs and shapes look right
    df, exp = make_clean_dataset()
    print("clean:", df.shape, exp)

    df, exp = make_within_split_duplicates()
    print("within-split dupes:", df.shape, exp)

    df, exp = make_cross_split_leakage()
    print("cross-split leakage:", df.shape, exp)

    df, exp = make_label_conflicts()
    print("label conflicts:", df.shape, exp)

    train_df, val_df, test_df, exp = make_three_way_split()
    print("three-way split:", train_df.shape, val_df.shape, test_df.shape, exp)

    a, b = make_mismatched_schema_pair()
    print("mismatched schema:", list(a.columns), list(b.columns))

    a, b = make_split_col_collision_pair()
    print("split col collision:", list(a.columns), list(b.columns))

    train_df, test_df, exp = make_realistic_mixed_scenario()
    print("realistic mixed:", train_df.shape, test_df.shape, exp)