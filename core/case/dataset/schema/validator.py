
from typing import Set

from core.case.dataset.schema import DataSchema
from core.case.dataset.schema import DataSchemaValidationReport
from core.case.dataset.common import DataSplitName
from core.case.dataset.schema.validation_models import DataSchemaErrorCode, DataSchemaWarningCode, WarningSeverity
class DataSchemaValidator:
    """
    Validates Dataset Schema
    """

    def validate(self, 
                 schema: DataSchema) -> DataSchemaValidationReport:
        
        report = DataSchemaValidationReport()
        self._validate_splits(schema, report)
        return report

    def _validate_splits (self,
                            schema: DataSchema,
                            report: DataSchemaValidationReport
                            ) -> None:
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

        reference_data_split= DataSplitName.TRAIN
        reference_features_set : Set[str] = set(schema.splits[reference_data_split].features.keys())
        for data_split_name, data_split_schema in schema.splits.items():
            if data_split_name == reference_data_split:
                continue
            current_features_set: Set[str] = set(data_split_schema.features.keys())
            
            if current_features_set != reference_features_set:
                missing_features = reference_features_set - current_features_set
                extra_features = current_features_set - reference_features_set
                if missing_features :
                    report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' is missing features found in '{reference_data_split.value}' : {list(missing_features)} ",
                        feature_names=list(missing_features),

                    )
                if extra_features :
                    report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' has extra features not found in '{reference_data_split.value}' : {list(extra_features)} ",
                        feature_names=list(extra_features),

                    )
    def _validate_identic_targets(self,
                                  schema: DataSchema,
                                  report: DataSchemaValidationReport)->None:
        reference_data_split= DataSplitName.TRAIN
        reference_targets = set(schema.splits[reference_data_split].targets.keys()) if schema.splits[reference_data_split].targets else None
        reference_has_targets = reference_targets is not None


        for data_split_name, data_split_schema in schema.splits.items():

            current_targets = set(data_split_schema.targets.keys()) if data_split_schema.targets else None
            current_has_targets = current_targets is not None

            if reference_has_targets!=current_has_targets:
                if reference_has_targets:
                    report.add_error(
                            code = DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                            detail= f"Target inconsistency: Reference split '{reference_data_split.value}' has targets "
                                        f"({sorted(list(reference_targets))}), but split '{data_split_name.value}' has no targets defined."
                
                                    )
                else:
                    report.add_error(
                        code= DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail= f"Target inconsistency: Reference split '{reference_data_split.value}' has no targets defined, "
                                f"but split '{data_split_name.value}' has targets ({sorted(list(current_targets))})."
                    )
            if reference_has_targets and current_has_targets:
                if reference_targets!=current_targets:
                    missing_targets = reference_targets - current_targets
                    extra_targets = current_targets - reference_targets
                    if missing_targets:
                        report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' is missing targets found in '{reference_data_split.value}' : {list(missing_targets)} ",
                        feature_names=list(missing_targets)
                        )
                    if extra_targets:
                        report.add_error(
                        code=DataSchemaErrorCode.INCONSISTENT_SCHEMA,
                        detail=f"Split '{data_split_name.value}' has extra targets not found in '{reference_data_split.value}' : {list(extra_targets)} ",
                        feature_names=list(extra_targets),

                    )
                    

        
        