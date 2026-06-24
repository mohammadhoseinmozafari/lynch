import polars as pl
import numpy as np

from core.observation.observation import ObservationBatch
from core.observation.type import ObservationType
from core.signal.processors.context import StructuralExecutionContext
from core.signal.signal import SignalTable
from core.signal.enums import SubjectType
from core.signal.processors.processor import SignalProcessor


class ColumnarMissingnessStructuralSignalProcessor(
    SignalProcessor[StructuralExecutionContext, SignalTable]
):
    """
    Processor for column-level structural signals.
    Requires ObservationType.COLUMN_MISSINGNESS (per-column missing rate, missing count, total count).
    Computes dataset-level statistics on the fly.
    """

    def __init__(self, config):
        self.config = config
        self.id = "columnar_missingness_structural_signal_processor"
        self.supporting_type = [ObservationType.COLUMN_MISSINGNESS]
        self.subject_type = SubjectType.FEATURE

    def run(self, context: StructuralExecutionContext) -> SignalTable:
        df: pl.DataFrame = context.batch.df

        # thresholds from config
        z_thresh = getattr(self.config, 'localized_feature_failure_z_threshold', 2.0)
        global_std_thresh = getattr(self.config, 'global_degradation_std_threshold', 0.02)
        global_mean_thresh = getattr(self.config, 'global_degradation_mean_threshold', 0.1)
        dominant_thresh = getattr(self.config, 'dominant_feature_failure_threshold', 0.5)

        # --- compute dataset statistics ---
        # mean and std of missing_rate across columns
        stats = df.select([
            pl.col("mean_missing_rate").alias("mean_missing_rate"),
            pl.col("std_missing_rate").alias("std_missing_rate"),
            pl.col("missing_count").sum().alias("total_missing_cells"),
        ])
        mean_missing = stats["mean_missing_rate"][0]
        std_missing = stats["std_missing_rate"][0]
        total_missing = stats["total_missing_cells"][0]

        # avoid division by zero
        if std_missing == 0:
            std_missing = 1e-9

        # --- per-column signals ---
        signal_df = df.with_columns([
            # z-score for each column
            ((pl.col("missing_rate") - mean_missing) / std_missing).alias("_z_score"),

            # localized feature failure: z > threshold
            (((pl.col("missing_rate") - mean_missing) / std_missing) > z_thresh)
            .cast(pl.Int8)
            .alias("localized_feature_failure"),

            # global feature degradation: dataset-level condition, constant for all columns
            pl.lit(int(std_missing < global_std_thresh and mean_missing > global_mean_thresh))
            .alias("global_feature_degradation"),

            # dominant feature failure: share of total missing cells
            (pl.col("missing_count") / total_missing).alias("dominant_feature_failure_ratio"),
            ((pl.col("missing_count") / total_missing) > dominant_thresh)
            .cast(pl.Int8)
            .alias("dominant_feature_failure"),

            # feature missingness outlier (IQR rule)
            # compute Q1, Q3, IQR on the fly using a window or expression? 
            # We'll do a separate computation.
        ])

        # IQR outlier detection
        q1 = df["missing_rate"].quantile(0.25)
        q3 = df["missing_rate"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        signal_df = signal_df.with_columns(
            (
                (pl.col("missing_rate") < lower) | (pl.col("missing_rate") > upper)
            )
            .cast(pl.Int8)
            .alias("feature_missingness_outlier")
        )

        # select final columns
        result = signal_df.select([
            "observation_id",
            "column_name",
            "localized_feature_failure",
            "global_feature_degradation",
            "dominant_feature_failure_ratio",
            "dominant_feature_failure",
            "feature_missingness_outlier",
        ])

        return SignalTable(result)


class RowsMissingnessStructuralSignalProcessor(
    SignalProcessor[StructuralExecutionContext, SignalTable]
):
    """
    Processor for row-level structural signals.
    Requires ObservationType.ROWS_MISSINGNESS (full_missing_rows_rate, high_missing_rows_rate).
    """

    def __init__(self, config):
        self.config = config
        self.id = "rows_missingness_structural_signal_processor"
        self.supporting_type = [ObservationType.ROWS_MISSINGNESS]
        self.subject_type = SubjectType.ROW

    def run(self, context: StructuralExecutionContext) -> SignalTable:
        df: pl.DataFrame = context.batch.df

        # thresholds
        complete_thresh = getattr(self.config, 'complete_row_corruption_threshold', 0.1)
        partial_thresh = getattr(self.config, 'partial_row_corruption_threshold', 0.2)
        mixed_full_thresh = getattr(self.config, 'mixed_row_corruption_full_threshold', 0.1)
        mixed_high_thresh = getattr(self.config, 'mixed_row_corruption_high_threshold', 0.2)

        signal_df = df.with_columns([
            (pl.col("full_missing_rows_rate") > complete_thresh)
            .cast(pl.Int8)
            .alias("complete_row_corruption"),

            (pl.col("high_missing_rows_rate") > partial_thresh)
            .cast(pl.Int8)
            .alias("partial_row_corruption"),

            (
                (pl.col("full_missing_rows_rate") > mixed_full_thresh) &
                (pl.col("high_missing_rows_rate") > mixed_high_thresh)
            )
            .cast(pl.Int8)
            .alias("mixed_row_corruption"),
        ])

        result = signal_df.select([
            "observation_id",
            "subject_name",
            "complete_row_corruption",
            "partial_row_corruption",
            "mixed_row_corruption",
        ])

        return SignalTable(result)


class RowsDistributionStructuralSignalProcessor(
    SignalProcessor[StructuralExecutionContext, SignalTable]
):
    """
    Processor for row‑distribution structural signals.
    Requires ObservationType.ROWS_MISSINGNESS_DISTRIBUTION (summary statistics per row‑missingness distribution).
    Assumes columns: std_missing_rate, mean_missing_rate, median_missing_rate,
                      p99_missing_rate, max_missing_rate, min_missing_rate.
    """

    def __init__(self, config):
        self.config = config
        self.id = "rows_distribution_structural_signal_processor"
        self.supporting_type = [ObservationType.ROWS_MISSINGNESS_DISTRIBUTION]
        self.subject_type = SubjectType.ROW_DISTRIBUTION

    def run(self, context: StructuralExecutionContext) -> SignalTable:
        df: pl.DataFrame = context.batch.df

        # thresholds
        std_low = getattr(self.config, 'uniform_row_std_threshold', 0.05)
        std_high = getattr(self.config, 'heterogeneous_row_std_threshold', 0.1)
        heavy_tail_thresh = getattr(self.config, 'heavy_tail_row_threshold', 0.3)
        skewed_thresh = getattr(self.config, 'skewed_row_threshold', 0.1)
        extreme_max_thresh = getattr(self.config, 'extreme_row_failure_max_threshold', 0.8)

        signal_df = df.with_columns([
            (pl.col("std_missing_rate") < std_low)
            .cast(pl.Int8)
            .alias("uniform_row_quality"),

            (pl.col("std_missing_rate") > std_high)
            .cast(pl.Int8)
            .alias("heterogeneous_row_quality"),

            ((pl.col("p99_missing_rate") - pl.col("median_missing_rate")) > heavy_tail_thresh)
            .cast(pl.Int8)
            .alias("heavy_tail_row_corruption"),

            ((pl.col("mean_missing_rate") - pl.col("median_missing_rate")).abs() > skewed_thresh)
            .cast(pl.Int8)
            .alias("skewed_row_corruption"),

            (pl.col("max_missing_rate") > extreme_max_thresh)
            .cast(pl.Int8)
            .alias("extreme_row_failure"),
        ])

        result = signal_df.select([
            "observation_id",
            "subject_name",
            "uniform_row_quality",
            "heterogeneous_row_quality",
            "heavy_tail_row_corruption",
            "skewed_row_corruption",
            "extreme_row_failure",
        ])

        return SignalTable(result)


class ColumnsDistributionStructuralSignalProcessor(
    SignalProcessor[StructuralExecutionContext, SignalTable]
):
    """
    Processor for column‑distribution structural signals.
    Requires ObservationType.COLUMNS_MISSINGNESS_DISTRIBUTION (summary statistics per column missingness distribution).
    Assumes columns: std_missing_rate, mean_missing_rate, median_missing_rate,
                      p99_missing_rate, max_missing_rate, min_missing_rate.
    """

    def __init__(self, config):
        self.config = config
        self.id = "columns_distribution_structural_signal_processor"
        self.supporting_type = [ObservationType.COLUMNS_MISSINGNESS_DISTRIBUTION]
        self.subject_type = SubjectType.COLUMN_DISTRIBUTION

    def run(self, context: StructuralExecutionContext) -> SignalTable:
        df: pl.DataFrame = context.batch.df

        # thresholds (reuse similar names but for columns)
        std_low = getattr(self.config, 'uniform_feature_std_threshold', 0.05)
        std_high = getattr(self.config, 'heterogeneous_feature_std_threshold', 0.1)
        heavy_tail_thresh = getattr(self.config, 'heavy_tail_feature_threshold', 0.3)
        skewed_thresh = getattr(self.config, 'skewed_feature_threshold', 0.1)
        extreme_max_thresh = getattr(self.config, 'extreme_feature_failure_max_threshold', 0.8)

        signal_df = df.with_columns([
            (pl.col("std_missing_rate") < std_low)
            .cast(pl.Int8)
            .alias("uniform_feature_quality"),

            (pl.col("std_missing_rate") > std_high)
            .cast(pl.Int8)
            .alias("heterogeneous_feature_quality"),

            ((pl.col("p99_missing_rate") - pl.col("median_missing_rate")) > heavy_tail_thresh)
            .cast(pl.Int8)
            .alias("heavy_tail_feature_corruption"),

            ((pl.col("mean_missing_rate") - pl.col("median_missing_rate")).abs() > skewed_thresh)
            .cast(pl.Int8)
            .alias("skewed_feature_corruption"),

            (pl.col("max_missing_rate") > extreme_max_thresh)
            .cast(pl.Int8)
            .alias("extreme_feature_failure"),
        ])

        result = signal_df.select([
            "observation_id",
            "subject_name",
            "uniform_feature_quality",
            "heterogeneous_feature_quality",
            "heavy_tail_feature_corruption",
            "skewed_feature_corruption",
            "extreme_feature_failure",
        ])

        return SignalTable(result)