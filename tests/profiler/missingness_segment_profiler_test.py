import numpy as np
import pandas as pd

from core.profilers.missingness.models import SegmentProfile
from core.profilers.missingness.segment_profiler import (
    PandasMissingnessSegmentProfiler,
)


def test_profile_segments_from_missing_value_column() -> None:
    df = pd.DataFrame(
        {
            "age": [18, 19, 20, 55, 57, 60],
            "country": ["us", "us", "ca", "ca", "ca", "ca"],
            "income": [10.0, 11.0, 12.0, 40.0, 41.0, 42.0],
            "target": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
        }
    )

    profiler = PandasMissingnessSegmentProfiler(max_depth=2, min_samples_leaf=1, random_state=7)
    segments = profiler.profile_segments(df, target="target")

    assert segments
    assert all(isinstance(segment, SegmentProfile) for segment in segments)

    total_rows = len(df)
    global_missing_rate = 0.5

    matching_segments = [
        segment
        for segment in segments
        if any("age" in rule for rule in segment.rules)
    ]
    assert matching_segments

    left_segment = min(segments, key=lambda segment: segment.missing_rate)
    right_segment = max(segments, key=lambda segment: segment.missing_rate)

    assert left_segment.row_count + right_segment.row_count == total_rows
    assert left_segment.global_missing_rate == global_missing_rate
    assert right_segment.global_missing_rate == global_missing_rate
    assert left_segment.coverage + right_segment.coverage == 1.0

    high_missing_segment = max(segments, key=lambda segment: segment.missing_rate)
    assert high_missing_segment.missing_rate == 1.0
    assert high_missing_segment.contribution == 1.0
    assert high_missing_segment.lift == 2.0
    assert any("age" in rule for rule in high_missing_segment.rules)


def test_profile_segments_from_boolean_mask() -> None:
    df = pd.DataFrame(
        {
            "age": [18, 19, 20, 55, 57, 60],
            "country": ["us", "us", "ca", "ca", "ca", "ca"],
        }
    )
    target_mask = pd.Series([True, True, True, False, False, False])

    profiler = PandasMissingnessSegmentProfiler(max_depth=2, min_samples_leaf=1, random_state=7)
    segments = profiler.profile_segments(df, target=target_mask)

    assert len(segments) >= 2
    assert sum(segment.row_count for segment in segments) == len(df)
    assert sum(segment.missing_count for segment in segments) == int(target_mask.sum())