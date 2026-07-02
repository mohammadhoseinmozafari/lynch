"""Pairwise correlation profiling for dataframe missingness masks."""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Any, Mapping

import numpy as np
import pandas as pd

from case.domain.value_objects.data_profile import DataProfile
from case.domain.value_objects.profile_namespace import ProfileNamespace
from core.profile import DatasetProfiler


class MissingnessCorrelationProfiler(DatasetProfiler):
    """Compute pairwise relationships between column missingness masks.

    The profiler is read-only with respect to both inputs. It consumes the
    column-rate namespace produced by the dependency resolver and returns a
    namespace for the engine to attach to the shared profile.
    """

    capability = "missingness.correlation"
    requires = {"profile.base", "missingness.column_rates"}
    provides = {capability}

    def profile(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
    ) -> ProfileNamespace:
        """Return pairwise binary-missingness metrics in deterministic order.
        """
        
        column_rates_namespace = profile.require_namespace("missingness.column_rates")
        base_name_space = profile.require_namespace("profile.base")

        
        counts = column_rates_namespace.require_metric("missing_count_by_column")
        total_rows = base_name_space.require_metric("total_rows")
        
        eligible_columns = list(df.columns)
        missing_matrix = df[eligible_columns].isna().to_numpy(dtype=np.uint8)

        accumulation_matrix = missing_matrix.astype(np.uint64, copy=False)
        both_missing = accumulation_matrix.T @ accumulation_matrix
        matrix_missing_counts = accumulation_matrix.sum(axis=0, dtype=np.uint64)

        missing_counts = np.asarray(
            [
                self._missing_count(
                    column,
                    counts,
                    int(matrix_missing_counts[index]),
                    total_rows,
                )
                for index, column in enumerate(eligible_columns)
            ],
            dtype=np.int64,
        )

        pairs = [
            self._build_pair(
                left_column=eligible_columns[left_index],
                right_column=eligible_columns[right_index],
                left_missing_count=int(missing_counts[left_index]),
                right_missing_count=int(missing_counts[right_index]),
                both_missing_count=int(both_missing[left_index, right_index]),
                total_rows=total_rows,
            )
            for left_index, right_index in combinations(
                range(len(eligible_columns)), 2
            )
        ]

        return ProfileNamespace(
            name=self.capability,
            metrics={
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "eligible_column_count": len(eligible_columns),
                "pair_count": len(pairs),
                "valid_pair_count": sum(pair["valid_pair"] for pair in pairs),
                "pairs": pairs,
            },
            artifacts={},
            metadata={
                "profiler": type(self).__name__,
                "method": "pairwise_binary_missingness",
            },
        )


    @staticmethod
    def _missing_count(
        column: Any,
        cached_counts: Mapping[Any, Any],
        fallback_count: int,
        total_rows: int,
    ) -> int:
        """Return and validate a cached count, or use the matrix-derived count."""
        raw_count = cached_counts.get(column, fallback_count)
        if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)):
            raise ValueError(f"Invalid missing count for column {column!r}")
        count = int(raw_count)
        if count < 0 or count > total_rows:
            raise ValueError(f"Invalid missing count for column {column!r}: {count}")
        return count

    @staticmethod
    def _build_pair(
        *,
        left_column: Any,
        right_column: Any,
        left_missing_count: int,
        right_missing_count: int,
        both_missing_count: int,
        total_rows: int,
    ) -> dict[str, Any]:
        """Build all MVP metrics for one pair with safe zero handling."""
        union_missing_count = (
            left_missing_count + right_missing_count - both_missing_count
        )
        left_missing_rate = (
            left_missing_count / total_rows if total_rows > 0 else 0.0
        )
        right_missing_rate = (
            right_missing_count / total_rows if total_rows > 0 else 0.0
        )
        support = both_missing_count / total_rows if total_rows > 0 else 0.0
        jaccard_similarity = (
            both_missing_count / union_missing_count
            if union_missing_count > 0
            else 0.0
        )
        p_right_given_left = (
            both_missing_count / left_missing_count
            if left_missing_count > 0
            else 0.0
        )
        p_left_given_right = (
            both_missing_count / right_missing_count
            if right_missing_count > 0
            else 0.0
        )
        lift_left_to_right = (
            p_right_given_left / right_missing_rate
            if right_missing_rate > 0
            else 0.0
        )
        lift_right_to_left = (
            p_left_given_right / left_missing_rate
            if left_missing_rate > 0
            else 0.0
        )

        phi_correlation = MissingnessCorrelationProfiler._phi_correlation(
            total_rows=total_rows,
            left_missing_count=left_missing_count,
            right_missing_count=right_missing_count,
            both_missing_count=both_missing_count,
        )
        invalid_reasons = []
        if total_rows == 0:
            invalid_reasons.append("dataset_has_no_rows")
        if left_missing_count == 0:
            invalid_reasons.append("left_column_has_no_missing_values")
        if right_missing_count == 0:
            invalid_reasons.append("right_column_has_no_missing_values")

        return {
            "left_column": left_column,
            "right_column": right_column,
            "left_missing_count": left_missing_count,
            "right_missing_count": right_missing_count,
            "both_missing_count": both_missing_count,
            "union_missing_count": union_missing_count,
            "left_missing_rate": left_missing_rate,
            "right_missing_rate": right_missing_rate,
            "support": support,
            "jaccard_similarity": jaccard_similarity,
            "phi_correlation": phi_correlation,
            "p_right_missing_given_left_missing": p_right_given_left,
            "p_left_missing_given_right_missing": p_left_given_right,
            "lift_left_to_right": lift_left_to_right,
            "lift_right_to_left": lift_right_to_left,
            "valid_pair": not invalid_reasons,
            "invalid_reason": "; ".join(invalid_reasons) or None,
        }

    @staticmethod
    def _phi_correlation(
        *,
        total_rows: int,
        left_missing_count: int,
        right_missing_count: int,
        both_missing_count: int,
    ) -> float | None:
        """Calculate phi from the 2x2 binary contingency table."""
        n11 = both_missing_count
        n10 = left_missing_count - n11
        n01 = right_missing_count - n11
        n00 = total_rows - n11 - n10 - n01
        denominator_product = (
            (n11 + n10)
            * (n01 + n00)
            * (n11 + n01)
            * (n10 + n00)
        )
        if denominator_product <= 0:
            return None
        return float((n11 * n00 - n10 * n01) / sqrt(denominator_product))
