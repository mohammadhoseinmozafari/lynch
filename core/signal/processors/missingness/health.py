import polars as pl
import numpy as np

from core.observation.observation import ObservationBatch
from core.observation.type import ObservationType
from core.signal.processors.context import HealthExecutionContext
from core.signal.signal import SignalTable
from core.signal.enums import SubjectType
from core.signal.processors.processor import SignalProcessor


class ColumnarMissingnessHealthProcessor(
    SignalProcessor[HealthExecutionContext,
                    SignalTable]
):

    def __init__(self, config):

        self.config = config
        self.id = "columnar_missingness_health_signal_processor"
        self.supporting_type = [ObservationType.COLUMN_MISSINGNESS]
        self.subject_type  = SubjectType.FEATURE

    def run(
        self,
        context : HealthExecutionContext
    ) -> SignalTable:

        df : pl.DataFrame = context.batch.df

        threshold = self.config.acceptable_missing_rate

        signal_df = (
            df

            .with_columns(

                [
                    (1 - pl.col("missing_rate"))
                    .alias("feature_completeness"),

                    pl.col("missing_rate")
                    .alias("feature_missingness"),

                    (
                        1
                        - pl.col("missing_rate")
                        / threshold
                    )
                    .clip(lower_bound=0.0, upper_bound=1.0)
                    .alias("feature_usability"),

                    (
                        1
                        - (
                            (
                                pl.col("missing_rate")
                                *
                                (
                                    1
                                    - pl.col("missing_rate")
                                )
                            )
                            /
                            pl.col("total_count")
                        )
                        .sqrt()
                    )
                    .clip(lower_bound=0.0, upper_bound=1.0)
                    .alias(
                        "feature_observation_confidence"
                    )

                ]

            )

            .select(

                [
                    "observation_id",

                    "column_name",

                    "reliability",

                    "feature_completeness",

                    "feature_missingness",

                    "feature_usability",

                    "feature_observation_confidence"

                ]

            )

        )

        return SignalTable(signal_df)
    



class RowsMissingnessHealthSignalProcessor(
    SignalProcessor[HealthExecutionContext,SignalTable]
    ):

    def __init__(self, config):
        self.config = config
        self.id = "rows_missingness_health_signal_processor"
        self.supporting_type = [ObservationType.ROWS_MISSINGNESS]
        self.subject_type  = SubjectType.ROW

    def run(self, context: HealthExecutionContext) -> SignalTable:

        df : pl.DataFrame = context.batch.df

        a = self.config.full_missing_row_integrity_param
        b = self.config.high_missing_row_integrity_param

        signal_df = df.with_columns(

            [
                # dataset row integrity
                (
                    1 - (
                        a * pl.col("full_missing_rows_rate")
                        +
                        b * pl.col("high_missing_rows_rate")
                    )
                )
                .clip(0.0, 1.0)
                .alias("dataset_row_integrity"),

                # complete row failure rate
                pl.col("full_missing_rows_rate")
                .alias("complete_row_failure_rate"),

                # partial row failure rate
                pl.col("high_missing_rows_rate")
                .alias("partial_row_failure_rate"),
            ]
        ).with_columns(

            [
                # derived health scores (same semantics as your old code)
                (1 - pl.col("complete_row_failure_rate"))
                .alias("complete_row_health"),

                (1 - pl.col("partial_row_failure_rate"))
                .alias("partial_row_health"),
            ]
        ).select(

            [
                "observation_id",
                "subject_name",

                "dataset_row_integrity",
                "complete_row_failure_rate",
                "partial_row_failure_rate",

                "complete_row_health",
                "partial_row_health",
            ]
        )

        return SignalTable(signal_df)
    


class DatasetMissingnessHealthSignalProcessor(
    SignalProcessor[HealthExecutionContext,SignalTable]
    ):


    def __init__(self, config):
        self.config = config
        self.id = "dataset_missingness_health_signal_processor"
        self.supporting_type = [ ObservationType.COLUMNS_MISSINGNESS_DISTRIBUTION ]
        self.subject_type  = SubjectType.DATASET


    def run(self, context: HealthExecutionContext) -> SignalTable:

        df = context.batch.df

        # expects one row per column summary observation
        # columns:
        # mean_missing_rate, min_missing_rate, max_missing_rate, p95, etc.

        signal_df = df.with_columns(

            [
                # dataset completeness
                (1 - pl.col("mean_missing_rate"))
                .alias("dataset_completeness"),

                # dataset missingness
                pl.col("mean_missing_rate")
                .alias("dataset_missingness"),

                # worst feature health
                (1 - pl.col("max_missing_rate"))
                .alias("worst_feature_health"),

                # best feature health
                (1 - pl.col("min_missing_rate"))
                .alias("best_feature_health"),
            ]
        ).select(

            [
                "observation_id",
                "subject_name",
                "dataset_completeness",
                "dataset_missingness",
                "worst_feature_health",
                "best_feature_health",
            ]
        )

        return SignalTable(signal_df)