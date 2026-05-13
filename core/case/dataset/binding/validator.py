from pydantic import BaseModel, Field

from core.case.dataset import DatasetBinding
from core.case.dataset.split import DataSplitName
from core.case.dataset.validation import DatasetValidationReport
from core.case.dataset.validation.models import DatasetErrorCode, DatasetWarningCode, WarningSeverity
from typing import Callable, Set, Dict



class DatasetBindingValidator :

    def validate(
            self,
            binding : DatasetBinding,
            report: DatasetValidationReport,


    ) -> DatasetValidationReport:
        
        report = DatasetValidationReport()
        self._validate_splits(binding, report)
        self._validate_drift_baseline(binding, report)
        self._validate_task_type(binding, report)

        self._validate_profiles (binding, report)
        

        return report




    def _validate_splits(self, binding: DatasetBinding, report: DatasetValidationReport) -> None:

        if not binding.get_split('train'):
            report.add_error(
                code=DatasetErrorCode.TRAIN_EMPTY,
                detail="Dataset doesn't have a 'train' split. "
                "Holmz needs to know what the model was trained on "
                "to produce valid explanations."
            )
    
        if not binding.get_split('val'):
            report.add_warning(
                code= DatasetWarningCode.VAL_EMPTY,
                detail= "Dataset doesn't contain a validation set. "
                "You can proceed without a val set, but explanation quality "
                "and calibration checks may be impacted.",
                severity=WarningSeverity.HIGH,

            )
        
        if not binding.get_split('test'):
            report.add_warning(
                code= DatasetWarningCode.TEST_EMPTY,
                detail= "The dataset doesn’t contain a 'test' split. " \
                "You can still run interpretability analyses on available splits," \
                " but generalization/performance-linked explanation checks may be limited without a dedicated test set.",
                severity=WarningSeverity.LOW
            )
        
        
         ### Validate whether features are identic across split schemas
        self._validate_identic_features(binding= binding, report=report)
        
        ### validate whether targets are identic across split schemas
        self._validate_identic_targets(binding= binding, report=report)

        
        self._validate_identic_feature_dtypes(binding= binding , report= report)


        self._validate_identic_feature_dtypes(binding=binding, report=report)
        self._validate_identic_target_dtypes(binding=binding , report = report)

        
        self._validate_split_fractions (binding = binding , report = report)


    def  _validate_drift_baseline(self, binding: DatasetBinding, report: DatasetValidationReport)-> None:
        drift_baseline = binding.drift_baseline
        if not drift_baseline:
            report.add_warning(
                code= DatasetWarningCode.DRIFT_BASELINE_EMPTY,
                detail = (
                    "No drift baseline was provided for the binding. "
                ),
                severity= WarningSeverity.LOW
            )
        else:
            if not drift_baseline in binding.data_splits.keys():
                report.add_error(
                    code= DatasetErrorCode.DRIFT_BASELINE_SPLIT_DOESNT_EXIST,
                    detail= (f"Split '{drift_baseline.value}' is set for drift baseline, but this split doesn't exist in the binding"
                    f"Available splits : {binding.get_split_names()}")
                )
    


    def _validate_task_type(self, binding : DatasetBinding, report: DatasetValidationReport)-> None:
        if not binding.task_type:
            report.add_warning(
                code= DatasetWarningCode.TASK_TYPE_EMPTY,
                detail= "task_type is missing – automatic explanation method selection may be suboptimal.",
                severity=WarningSeverity.LOW
            )



    def _validate_profiles(self,
                           binding: DatasetBinding,
                           report : DatasetValidationReport)-> None:
            
            ###Every Feature in the Schema Must Have a Profile
            
            for split in binding.data_splits.values():
                if split.data_profile:
                    schema_features = split.data_schema.get_features_names()
                    profile_features = split.data_profile.get_feature_names()

                    missing_features= schema_features- profile_features
                    extra_features = profile_features - schema_features

                    if missing_features:
                        report.add_error(
                            code= DatasetErrorCode.FEATURE_PROFILE_MISMATCH,
                            detail= f"Split '{split.name}': Missing profiles for features: {missing_features}"
                        )
                    if extra_features:
                        report.add_error(
                            code= DatasetErrorCode.FEATURE_PROFILE_MISMATCH,
                            detail= f"Split '{split.name}': Extra profiles for non-existent features: {extra_features}"
                        )
                        

                else:
                    report.add_warning(
                        code= DatasetWarningCode.DATA_PROFILE_EMPTY,
                        detail=(f"Split '{split.name}' of the dataset doesn't have data profile. Explanation quality might be impacted")
                        ,
                        severity=WarningSeverity.HIGH

                    )
            



        


    def _validate_identic_features(self, 
                                   binding:DatasetBinding, 
                                   report: DatasetValidationReport)->None:
        """
        Ensures that all data splits expose exactly the same set of features.

        The 'train' split is used as the reference. Every other split is compared against it.
        If a split is missing features present in train, or has extra features not in train,
        a blocking error is added to the report.

        Why this matters:
        Explanation methods like SHAP and LIME assume the model sees the same features
        during training and inference. If splits have different feature sets, explanations
        generated on one split may be invalid for another. Counterfactual generation depends
        on knowing which features are available across all splits.

        Parameters:
            binding: The DatasetBinding containing all splits to validate.
            report: The DatasetValidationReport to add errors to.

        """

        reference_data_split= binding.get_split('train')
        assert reference_data_split
        reference_features_set : Set[str] = reference_data_split.data_schema.get_features_names()

        for data_split_name, data_split in binding.data_splits.items():
            if data_split_name == reference_data_split:
                continue
            current_features_set: Set[str] = data_split.data_schema.get_features_names()
            
            if current_features_set != reference_features_set:
                missing_features = reference_features_set - current_features_set
                extra_features = current_features_set - reference_features_set
                if missing_features :
                    report.add_error(
                        code=DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' is missing features found in '{reference_data_split}' : {list(missing_features)} ",
                        feature_names=list(missing_features),

                    )
                if extra_features :
                    report.add_error(
                        code=DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' has extra features not found in '{reference_data_split}' : {list(extra_features)} ",
                        feature_names=list(extra_features),

                    )
    
    def _validate_identic_targets(self,
                                  binding: DatasetBinding,
                                  report: DatasetValidationReport)->None:
        """
        Ensures that all data splits define the same set of target columns.

        The 'train' split is used as the reference. Every other split is compared against it.
        Three types of inconsistency are detected:

        1. Train has targets but another split does not.
        2. Train has no targets but another split does.
        3. Both have targets but the sets differ.

        All three produce blocking errors because target inconsistency means the model's
        prediction task is ambiguous across splits. This would cause explanation methods
        to produce meaningless results, especially for counterfactual and contrastive
        explanation types.

        Parameters:
            schema: The DataSchema containing all splits to validate.
            report: The DataSchemaValidationReport to add errors to.

        Errors produced:
            INCONSISTENT_SCHEMA: Targets differ between splits. The error detail lists
                which targets are missing, extra, or the type of presence inconsistency.
        """
        reference_data_split= binding.get_split('train')
        assert reference_data_split

        reference_targets :Set[str]| None= reference_data_split.data_schema.get_targets_names()
        reference_has_targets = reference_targets is not None


        for data_split_name, data_split in binding.data_splits.items():
            
            current_targets = data_split.data_schema.get_targets_names()
            current_has_targets = current_targets is not None

            if reference_has_targets!=current_has_targets:
                if reference_has_targets:
                    report.add_error(
                            code = DatasetErrorCode.INCONSISTENT_SCHEMA,
                            detail= f"Target inconsistency: Reference split '{reference_data_split.name}' has targets " 
                                        f"({sorted(list(reference_targets))}), but split '{data_split_name.value}' has no targets defined."
                
                                    )
                else:
                    assert current_targets

                    report.add_error(
                        code= DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail= f"Target inconsistency: Reference split '{reference_data_split.name}' has no targets defined, " 
                                f"but split '{data_split_name.value}' has targets ({sorted(list(current_targets))})." 
                    )
            if reference_has_targets and current_has_targets:
                if reference_targets != current_targets:
                    missing_targets = reference_targets - current_targets
                    extra_targets = current_targets - reference_targets
                    if missing_targets:
                        report.add_error(
                        code=DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' is missing targets found in '{reference_data_split.name}' : {list(missing_targets)} ", 
                        feature_names=list(missing_targets)
                        )
                    if extra_targets:
                        report.add_error(
                        code=DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' has extra targets not found in '{reference_data_split.name}' : {list(extra_targets)} ", 
                        feature_names=list(extra_targets),

                    )
    
    def _validate_identic_feature_dtypes(
    self,
    binding: DatasetBinding,
    report: DatasetValidationReport,
) -> None:
        """
        Ensures that features with the same name across splits have identical dtypes
        and consistent expected_range types.

        The 'train' split is the reference. For every other split, each feature
        present in both is checked:
            - `dtype` must be exactly equal.
            - If both sides have an `expected_range`, their `feature_type` must match.

        A mismatch would mean a column has a different interpretation in different
        splits, making explanations unreliable.

        Parameters:
            binding: The DatasetBinding containing the splits.
            report:   The DatasetValidationReport to add errors to.
        """
        reference_split = binding.get_split("train")
        assert reference_split

        reference_features = reference_split.data_schema.features

        for split_name, split in binding.data_splits.items():
            if split.name == "train":
                continue

            current_features = split.data_schema.features
            
            common_feature_names = set(reference_features.keys())
            for feature_name in common_feature_names:
                ref_feat = reference_features[feature_name]
                cur_feat = current_features[feature_name]

                # --- dtype comparison ---
                if ref_feat.dtype != cur_feat.dtype:
                    report.add_error(
                        code=DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail=(
                            f"Feature '{feature_name}' has inconsistent dtype across splits: "
                            f"'{split_name.value}' has {cur_feat.dtype.value}, "
                            f"but 'train' has {ref_feat.dtype.value}."
                        ),
                        feature_names=[feature_name],
                    )


    def _validate_identic_target_dtypes(
    self,
    binding: DatasetBinding,
    report: DatasetValidationReport,
) -> None:
        """
        Ensures that targets with the same name across splits have identical dtypes
        and consistent expected_range types.

        The 'train' split is the reference. For every other split, each target
        present in both is checked:
            - `dtype` must be exactly equal.
            - If both sides have an `expected_range`, their `feature_type` must match.

        A mismatch would mean a column has a different interpretation in different
        splits, making explanations unreliable.

        Parameters:
            binding: The DatasetBinding containing the splits.
            report:   The DatasetValidationReport to add errors to.
        """
        reference_split = binding.get_split("train")
        assert reference_split

        reference_targets = reference_split.data_schema.targets
        assert reference_targets

        for split_name, split in binding.data_splits.items():
            if split.name == "train":
                continue

            current_targets = split.data_schema.targets
            assert current_targets
            
            common_target_names = set(reference_targets.keys())

            for target_name in common_target_names:
                ref_targ = reference_targets[target_name]
                cur_targ = current_targets[target_name]

                # --- dtype comparison ---
                if ref_targ.dtype != cur_targ.dtype:
                    report.add_error(
                        code=DatasetErrorCode.INCONSISTENT_SCHEMA,
                        detail=(
                            f"Target '{target_name}' has inconsistent dtype across splits: "
                            f"'{split_name.value}' has {cur_targ.dtype.value}, "
                            f"but 'train' has {ref_targ.dtype.value}."
                        ),
                        feature_names=[target_name],
                    )


    def _validate_split_fractions(self, binding : DatasetBinding, report : DatasetValidationReport) -> None:
        
        missing_split_fractions =[]
        fractions_sum = 0.0
        expected_sum = 1.0
        tolerance = 0.05
        for split in binding.data_splits.values():
            if split.split_fraction is None:
                missing_split_fractions.append(split.name)
            else:
                fractions_sum+= split.split_fraction

        all_have_fractions = len(missing_split_fractions) == 0

        if all_have_fractions:
            if abs(fractions_sum - expected_sum) > tolerance:
                report.add_warning(
                    code = DatasetWarningCode.INVALID_SPLIT_FRACTIONS_SUM,
                    detail=(
                        f"Sum of split fractions ({fractions_sum:.6f}) does not equal "
                        f"expected value ({expected_sum}) within tolerance ({tolerance}), this might indicate problem in data pipeline, or data leakage."
                    )
                    ,
                    severity=WarningSeverity.HIGH
                )
        else: 
            report.add_warning(
                code = DatasetWarningCode.SPLIT_FRACTIONS_NOT_PROVIDED,
                detail = (f"Missing split_fraction in {len(missing_split_fractions)} splits: {missing_split_fractions}"
                          f"Sum of split fractions is ({fractions_sum:.6f})"),
                severity=WarningSeverity.MEDIUM


            )
