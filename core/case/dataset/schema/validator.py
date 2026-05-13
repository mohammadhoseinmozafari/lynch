


from core.case.dataset.schema import DataSchema, DataSchemaValidationReport 
from core.case.dataset.schema.validation_models import DataSchemaErrorCode, DataSchemaWarningCode, WarningSeverity
from typing import Set
class DataSchemaValidator:
    """
    DataSchemaValidator validates a DataSchema to ensure it meets the minimum requirements
    for trustworthy model explanations. It checks that required splits are present, that
    feature sets are identical across all splits, and that target definitions are consistent.

    A valid schema must have at minimum a 'train' split. The 'val' and 'test' splits are
    strongly recommended but not required. All splits must expose the same set of features
    and targets. Inconsistencies between splits will cause explanation generation to fail
    or produce misleading results.

    Validation produces a DataSchemaValidationReport containing errors (which block model
    registration) and warnings (which allow registration but flag potential issues).
    """

    def validate(self, 
                 schema: DataSchema) -> DataSchemaValidationReport:
        """
        Validates the entire DataSchema and returns a DataSchemaValidationReport.

        The validation runs multiple checks in sequence:
        1. Required splits are present (train required, val and test recommended)
        2. Feature names are identical across all splits
        3. Target definitions are identical across all splits

        Parameters:
            schema: The DataSchema to validate. Must contain at least a 'train' split.

        Returns:
            DataSchemaValidationReport containing any errors or warnings found.
            An empty errors list means the schema passed all validation checks.
        """
        
        report = DataSchemaValidationReport()
        self._validate_splits(schema, report)
        return report

    def _validate_splits (self,
                            schema: DataSchema,
                            report: DataSchemaValidationReport
                            ) -> None:
        """
        Checks that required and recommended data splits are present in the schema.

        A 'train' split is mandatory. Without it, Holmz cannot determine what data
        the model was trained on, which makes explanations unreliable. Missing train
        split produces a blocking error.

        A 'val' split is strongly recommended. Without it, explanation quality checks
        and calibration analyses cannot be performed. Missing val split produces a
        high-severity warning but does not block registration.

        A 'test' split is recommended for completeness. Without it, generalization-aware
        explanation checks are limited. Missing test split produces a low-severity warning.

        After checking split presence, this method delegates to:
        - _validate_identic_features: ensures all splits have the same features
        - _validate_identic_targets: ensures all splits have the same targets
        """
        if not schema.get_split('train'):
            report.add_error(
                code=DataSchemaErrorCode.TRAIN_EMPTY,
                detail="Dataset doesn't have a 'train' split. "
                "Holmz needs to know what the model was trained on "
                "to produce valid explanations."
            )
        
        if not schema.get_split('val'):
            report.add_warning(
                code= DataSchemaWarningCode.VAL_EMPTY,
                detail= "Dataset doesn't contain a validation set. "
                "You can proceed without a val set, but explanation quality "
                "and calibration checks may be impacted.",
                severity=WarningSeverity.HIGH,

            )
        
        if not schema.get_split('test'):
            report.add_warning(
                code= DataSchemaWarningCode.TEST_EMPTY,
                detail= "The dataset doesn’t contain a 'test' split. " \
                "You can still run interpretability analyses on available splits," \
                " but generalization/performance-linked explanation checks may be limited without a dedicated test set.",
                severity=WarningSeverity.LOW
            )
        ### Validate whether features are identic across split schemas
        self._validate_identic_features(schema=schema, report=report)
        ### validate whether targets are identic across split schemas
        self._validate_identic_targets(schema=schema, report=report)

        
            
    def _validate_identic_features(self, 
                                   schema:DataSchema, 
                                   report: DataSchemaValidationReport)->None:
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
            schema: The DataSchema containing all splits to validate.
            report: The DataSchemaValidationReport to add errors to.

        Errors produced:
            INCONSISTENT_SCHEMA: A split has different features than the train split.
                The error detail lists which features are missing or extra.
        """

        reference_data_split= 'train'
        reference_features_set : Set[str] = schema.get_split(reference_data_split).get_features_names() # type: ignore
        for data_split_name, data_split_schema in schema.splits.items():
            if data_split_name == reference_data_split:
                continue
            current_features_set: Set[str] = data_split_schema.get_features_names()
            
            if current_features_set != reference_features_set:
                missing_features = reference_features_set - current_features_set
                extra_features = current_features_set - reference_features_set
                if missing_features :
                    report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' is missing features found in '{reference_data_split}' : {list(missing_features)} ",
                        feature_names=list(missing_features),

                    )
                if extra_features :
                    report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' has extra features not found in '{reference_data_split}' : {list(extra_features)} ",
                        feature_names=list(extra_features),

                    )
    def _validate_identic_targets(self,
                                  schema: DataSchema,
                                  report: DataSchemaValidationReport)->None:
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
        reference_data_split= schema.get_split('train')
        reference_targets = reference_data_split.get_targets_names() if reference_data_split else None
        
        reference_has_targets = reference_targets is not None


        for data_split_name, data_split_schema in schema.get_splits().items():
            
            current_targets = data_split_schema.get_targets_names()
            current_has_targets = current_targets is not None

            if reference_has_targets!=current_has_targets:
                if reference_has_targets:
                    report.add_error(
                            code = DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                            detail= f"Target inconsistency: Reference split '{reference_data_split.get_split_name()}' has targets " # type: ignore
                                        f"({sorted(list(reference_targets))}), but split '{data_split_name.value}' has no targets defined."
                
                                    )
                else:
                    report.add_error(
                        code= DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail= f"Target inconsistency: Reference split '{reference_data_split.get_split_name()}' has no targets defined, " # type: ignore
                                f"but split '{data_split_name.value}' has targets ({sorted(list(current_targets))})." # type: ignore
                    )
            if reference_has_targets and current_has_targets:
                if reference_targets!=current_targets:
                    missing_targets = reference_targets - current_targets
                    extra_targets = current_targets - reference_targets
                    if missing_targets:
                        report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' is missing targets found in '{reference_data_split.get_split_name()}' : {list(missing_targets)} ", # type: ignore
                        feature_names=list(missing_targets)
                        )
                    if extra_targets:
                        report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' has extra targets not found in '{reference_data_split.get_split_name()}' : {list(extra_targets)} ", # type: ignore
                        feature_names=list(extra_targets),

                    )
                    

        
        
