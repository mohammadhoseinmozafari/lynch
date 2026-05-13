from __future__ import annotations
from core.case.dataset.profile import DataProfile, DataProfileValidationReport, NumericStats, DataProfileErrorCode, DataProfileWarningCode
from core.case.dataset.profile.models import CategoricalStats

    

class DataProfileValidator:
    """
    Validates Data Profile
    """

    def validate(
        self,
        profile: DataProfile
    ) -> DataProfileValidationReport:
        report = DataProfileValidationReport()



        for  split_name, data_split in profile.splits.items():
            for feature_name, feature_profile in data_split.feature_profiles.items():
                feature_stats  = feature_profile.feature_stats
                if not feature_stats:
                    report.add_warning(
                        code= DataProfileWarningCode.STATS_MISSING,
                        detail=f"Feature '{feature_name}' in split '{split_name.value}' has no statistics.",
                        feature_name= feature_name
                    )
                
                else:
                
                    if isinstance(feature_stats, NumericStats):
                        self._validate_numeric_stats_consistency(
                            feature_name=feature_name,
                            stats=feature_stats,
                            report = report
                        )

                    elif isinstance(feature_stats, CategoricalStats):
                        self._validate_categorical_stats_consistency(
                            feature_name=feature_name,
                            stats=feature_stats,
                            report=report
                        )
        
        self._validate_drift_baseline_exists(profile=profile, report= report)
        
        return report
                        


                    

            
    


    def _validate_numeric_stats_consistency(
        self,
        feature_name: str,
        stats: NumericStats,
        report: DataProfileValidationReport
        ) -> None:
        """Validates internal consistency of NumericStats."""
        if stats.range  and stats.mean:
            numeric_range = stats.range
            if numeric_range.min is not None:
                if stats.mean < numeric_range.min:
                    report.add_error(
                        code=  DataProfileErrorCode.INCONSISTENT_NUMERIC_STATS,
                        detail = f"Feature '{feature_name}' : mean ({stats.mean}) has smaller value than the min value ({numeric_range.min})"
                        f"This indicates corrupted statistics or a data pipeline error.",
                        feature_name=feature_name

                    )
            if numeric_range.max is not None:
                if stats.mean > numeric_range.max :
                    report.add_error(
                        code=  DataProfileErrorCode.INCONSISTENT_NUMERIC_STATS,
                        detail = f"Feature '{feature_name}' : mean ({stats.mean}) has bigger value than the max value ({numeric_range.max})"
                        f"This indicates corrupted statistics or a data pipeline error.",
                        feature_name=feature_name

                    )


            
        # Validate quartiles fall within range
        if stats.range and stats.quartiles:
            numeric_range = stats.range
            q1, q2, q3 = stats.quartiles
            if numeric_range.min:
                if q1 < numeric_range.min or q2 < numeric_range.min or q3 < numeric_range.min:
                    report.add_error(
                        code=DataProfileErrorCode.INCONSISTENT_NUMERIC_STATS,
                        detail=(
                            f"Feature '{feature_name}': one or more quartiles "
                            f"(Q1={q1}, Q2={q2}, Q3={q3}) are below the minimum "
                            f"expected value ({numeric_range.min}). "
                            f"All quartiles must fall within the feature's range."
                        ),
                        feature_name=feature_name
                    )
            if numeric_range.max:
                if q1 > numeric_range.max or q2 > numeric_range.max or q3 > numeric_range.max:
                    report.add_error(
                        code=DataProfileErrorCode.INCONSISTENT_NUMERIC_STATS,
                        detail=(
                            f"Feature '{feature_name}': one or more quartiles "
                            f"(Q1={q1}, Q2={q2}, Q3={q3}) are above the maximum "
                            f"expected value ({numeric_range.max}). "
                            f"All quartiles must fall within the feature's range."
                        ),
                        feature_name = feature_name
                    )
        
        # Validate quartile ordering: Q1 ≤ Q2 ≤ Q3
        if stats.quartiles:
            q1, q2, q3 = stats.quartiles
            if not (q1 <= q2 <= q3):
                report.add_error(
                    code=DataProfileErrorCode.INCONSISTENT_NUMERIC_STATS,
                    detail=(
                        f"Feature '{feature_name}': quartiles are not properly ordered. "
                        f"Expected Q1 ({q1}) ≤ Q2 ({q2}) ≤ Q3 ({q3}). "
                        f"The quartiles must be in non-decreasing order."
                    ),
                    feature_name=feature_name
                )


        if stats.std  :
            if stats.std == 0.0:
                report.add_warning(
                    code = DataProfileWarningCode.ZERO_VARIANCE_FEATURE,
                    detail=(
                        f"Feature '{feature_name}' has STD of zero. "
                        
                    ),
                    feature_name=feature_name,

                )
            elif stats.std < 0.0 :
                report.add_warning(
                    code = DataProfileWarningCode.NEGATIVE_VARIANCE_FEATURE,
                    detail=(
                        f"Feature '{feature_name}' has negative STD. "
                        
                    ),
                    feature_name=feature_name,

                )
            
            
    # Warn if mean is provided but no range or quartiles for context
        if stats.mean and not stats.range and not stats.quartiles:
                report.add_warning(
                    code=DataProfileWarningCode.STATS_MISSING,
                    detail=(
                        f"Feature '{feature_name}': mean ({stats.mean}) is provided "
                        f"without an expected range or quartiles. Explanation baselines "
                        f"will rely solely on the mean, which may not represent the full "
                        f"distribution. Consider providing range or quartiles for more "
                        f"accurate drift detection and counterfactual constraints."
                    ),
                    feature_name=feature_name
                )
        
    
    # Validate that if range is provided, min ≤ max 
        if stats.range:
            numeric_range = stats.range
            if numeric_range.min is not None and numeric_range.max is not None and numeric_range.min > numeric_range.max:
                report.add_error(
                    code=DataProfileErrorCode.INCONSISTENT_NUMERIC_STATS,
                    detail=(
                        f"Feature '{feature_name}': range min ({numeric_range.min}) is greater "
                        f"than range max ({numeric_range.max}). The minimum must be less than "
                        f"or equal to the maximum."
                    ),
                    feature_name=feature_name
                )
    def _validate_categorical_stats_consistency(
            self,
            feature_name : str,
            stats: CategoricalStats,
            report: DataProfileValidationReport
    ) -> None:
        
        if stats.top_values:
            top_values = stats.top_values

            if stats.cardinality and len(top_values)> stats.cardinality:
                report.add_error(
                    code= DataProfileErrorCode.INCONSISTENT_CATEGORICAL_STATS,
                    detail=(
                    f"Feature '{feature_name}': {len(top_values)} top_values provided "
                    f"but cardinality is only {stats.cardinality}. "
                    f"More unique values listed than the feature actually has."
                ),
                feature_name=feature_name
            )
                
            invalid_frequencies ={
            value: freq 
            for value, freq in top_values.items() if freq<0.0 or freq>1.0
            }
            if invalid_frequencies:
                report.add_error(
                    code=DataProfileErrorCode.INCONSISTENT_CATEGORICAL_STATS,
                    detail=(
                    f"Feature '{feature_name}': top_values contain frequencies "
                    f"outside [0.0, 1.0]: {invalid_frequencies}. "
                    f"Frequencies must be between 0 and 1."
                ),
                feature_name=feature_name
                )
            
            total_frequency = sum(top_values.values())
            if total_frequency> 1.0 :
                report.add_error(
                    code=DataProfileErrorCode.INCONSISTENT_CATEGORICAL_STATS,
                    detail=(
                    f"Feature '{feature_name}': top_values frequencies sum to "
                    f"{total_frequency:.4f}, which exceeds 1.0. "
                    f"Frequencies should sum to at most 1.0 (they represent a subset)."
                ),
                feature_name=feature_name

                )
            
            if stats.cardinality and total_frequency < 0.3 :
                report.add_warning(
                code=DataProfileWarningCode.LOW_TOP_VALUES_COVERAGE,
                detail=(
                    f"Feature '{feature_name}': top_values cover only "
                    f"{total_frequency:.1%} of the data (cardinality={stats.cardinality}). "
                    f"The most frequent categories are underrepresented. "
                    f"Consider providing more top values for better explanation coverage."
                ),
                feature_name=feature_name
            )
    
    def _validate_drift_baseline_exists(
            self,
            profile : DataProfile,
            report : DataProfileValidationReport
    ) -> None:
        
        baseline_split_name = profile.drift_baseline

        if baseline_split_name:
            split_names = [name.value for name in profile.splits.keys()]
            baseline_exists = baseline_split_name.value in split_names

            if not baseline_exists:
                report.add_error(

                    code=DataProfileErrorCode.DRIFT_BASELINE_NOT_FOUND,
            detail=(
                f"Drift baseline references split '{baseline_split_name.value}', "
                f"but this split does not exist in the profile. "
                f"Available splits: {[name.value for name in profile.splits.keys()]}. "
                f"Either add a profile for '{baseline_split_name.value}' or "
                f"change drift_baseline to one of the available splits, or set it to None."
                )
                )
            