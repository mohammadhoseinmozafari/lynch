# Holmes Platform — Module & Class Specification

> Architecture: Event-Driven Hexagonal | Pattern: Ports & Adapters + Capability Registry  
> Package root: `holmes.*`

---

## Module Index

| Module | Layer | Description |
|---|---|---|
| `holmes.core` | Domain | Shared domain primitives used across all modules |
| `holmes.ports` | Ports | Abstract interface contracts (never change) |
| `holmes.events` | Domain | Domain events and in-process event bus |
| `holmes.application` | Application | Commands, handlers, orchestration, capability registry |
| `holmes.mems` | Shared Kernel | Model Entity Metadata Store |
| `holmes.iss` | Shared Kernel | Inference & Simulation Service |
| `holmes.eos` | Shared Kernel | Explanation & Observability Service |
| `holmes.edas` | Shared Kernel | Evidence & Diagnostics Artifact Store |
| `holmes.data_investigation` | Domain | Data Investigation Engine (6 subsystems) |
| `holmes.model_investigation` | Domain | Model Investigation Engine (7 subsystems) |
| `holmes.system_investigation` | Domain | System Investigation Engine (5 subsystems) |
| `holmes.adapters.*` | Adapters | All pluggable adapter implementations |
| `holmes.api` | Interface | REST / GraphQL API layer |
| `holmes.sdk` | Interface | Python SDK for ML developers |

---

## 1. `holmes.core`

Shared domain primitives. No external dependencies. Every other module depends on this one.

### Classes

#### `Finding`
The atomic domain primitive. Every investigation subsystem emits `Finding` objects.
- **Attributes:** `id: UUID`, `type: FindingType`, `severity: Severity`, `confidence: float`, `title: str`, `description: str`, `evidence: list[Evidence]`, `recommendations: list[Recommendation]`, `model_id: str | None`, `dataset_id: str | None`, `segment: dict | None`, `created_at: datetime`, `tier: CapabilityTier`
- **Methods:** `with_evidence(e: Evidence) -> Finding`, `with_recommendation(r: Recommendation) -> Finding`, `to_dict() -> dict`, `validate() -> bool`

#### `FindingType` (enum)
All possible finding type identifiers across all three investigation engines.
- Data: `MISSING_NOT_AT_RANDOM`, `DUPLICATE_SPIKE`, `OUTLIER_CLUSTER`, `SCHEMA_DRIFT`, `STALE_DATA`, `TEMPORAL_LEAKAGE`, `PREPROCESSING_LEAKAGE`, `PROXY_VARIABLE`, `TRAIN_TEST_OVERLAP`, `MISLABELED_INSTANCE`, `ANNOTATOR_DISAGREEMENT`, `LABEL_DRIFT`, `SYSTEMATIC_NOISE`, `POPULATION_GAP`, `INTERSECTIONAL_GAP`, `EXTRAPOLATION_ZONE`, `LOW_SIGNAL`, `FEATURE_REDUNDANCY`, `BAYES_ERROR_ESTIMATE`, `DATA_LIMITED`
- Model: `ROBUSTNESS_PROFILE`, `ADVERSARIAL_VULNERABILITY`, `OOD_BOUNDARY`, `FAILURE_SLICE`, `EMERGENT_FAILURE`, `BIAS_INDICATOR`, `CALIBRATION_ERROR`, `OVERCONFIDENCE`, `UNDERCONFIDENCE`, `SHORTCUT_FEATURE`, `SPURIOUS_CORRELATION`, `FRAGILE_PREDICTION`, `UNSTABLE_EXPLANATION`, `FEATURE_INTERACTION`, `SUPPRESSION_EFFECT`
- System: `DATA_DRIFT`, `PREDICTION_DRIFT`, `PERFORMANCE_DEGRADATION`, `CONFIDENCE_SHIFT`, `ROOT_CAUSE_IDENTIFIED`, `PERFORMANCE_ATTRIBUTION`, `PERFORMANCE_CEILING`, `RECOMMENDED_ACTION`, `AUTOMATED_FIX`

#### `Severity` (enum)
- Values: `LOW = 1`, `MEDIUM = 2`, `HIGH = 3`, `CRITICAL = 4`
- **Methods:** `from_score(score: float) -> Severity`

#### `CapabilityTier` (enum)
- Values: `OSS`, `PRO`

#### `Evidence`
Supporting data attached to a finding.
- **Attributes:** `type: EvidenceType`, `payload: dict`, `samples: list[dict] | None`, `statistics: dict | None`, `chart_data: dict | None`
- **Methods:** `from_dataframe(df) -> Evidence`, `from_stats(stats: dict) -> Evidence`

#### `EvidenceType` (enum)
- Values: `STATISTICAL_TEST`, `SAMPLE_ROWS`, `DISTRIBUTION_COMPARISON`, `TIME_SERIES`, `FEATURE_IMPORTANCE`, `CONFUSION_MATRIX`, `RELIABILITY_DIAGRAM`

#### `Recommendation`
An actionable suggestion attached to a finding.
- **Attributes:** `action: str`, `rationale: str`, `expected_impact: str`, `effort: EffortLevel`, `priority: int`, `estimated_metric_gain: float | None`

#### `EffortLevel` (enum)
- Values: `LOW`, `MEDIUM`, `HIGH`

#### `Report` (abstract base class)
Base for all health reports produced by investigation engines.
- **Attributes:** `id: UUID`, `engine: str`, `findings: list[Finding]`, `health_score: HealthScore`, `recommendations: list[Recommendation]`, `generated_at: datetime`, `model_id: str | None`, `dataset_id: str | None`
- **Methods:** `top_findings(n: int) -> list[Finding]`, `critical_findings() -> list[Finding]`, `to_dict() -> dict`

#### `HealthScore`
A 0–100 score with breakdown by dimension.
- **Attributes:** `overall: float`, `breakdown: dict[str, float]`, `trend: float | None`
- **Methods:** `grade() -> str`, `is_healthy(threshold: float = 70.0) -> bool`

#### `InvestigationContext`
Immutable context object passed into every subsystem during a run.
- **Attributes:** `run_id: UUID`, `model_id: str | None`, `dataset_id: str | None`, `config: dict`, `tier: CapabilityTier`, `triggered_by: str`
- **Methods:** `with_config(key: str, value) -> InvestigationContext`

#### `DataArtifactPointer`
A reference to externally stored data without loading it into memory.
- **Attributes:** `uri: str`, `format: DataFormat`, `size_bytes: int | None`, `schema_id: str | None`
- **Methods:** `resolve(storage: IStorageBackend) -> Any`

#### `DataFormat` (enum)
- Values: `PARQUET`, `CSV`, `JSON`, `ARROW`, `DELTA`

---

## 2. `holmes.ports`

Abstract interface contracts. The domain depends only on these. All adapters implement these.

### Classes

#### `IModelStore` (abstract)
Port for all model registry operations.
- `register_model(model: ModelVersion) -> str`
- `get_model_version(model_id: str, version: str) -> ModelVersion`
- `list_model_versions(model_id: str) -> list[ModelVersion]`
- `get_latest_version(model_id: str) -> ModelVersion`
- `get_training_run(run_id: str) -> TrainingRun`
- `save_model_artifact(model_id: str, artifact: bytes, metadata: dict) -> str`

#### `IDataCatalog` (abstract)
Port for all dataset and data catalog operations.
- `register_dataset(binding: DatasetBinding) -> str`
- `get_dataset(dataset_id: str) -> DatasetBinding`
- `get_data_profile(dataset_id: str) -> DataProfile`
- `get_schema(dataset_id: str) -> DataSchema`
- `load_split(dataset_id: str, split: str) -> DataArtifactPointer`
- `list_datasets(tags: dict | None) -> list[DatasetBinding]`

#### `IEventBus` (abstract)
Port for publish/subscribe event communication between engines.
- `publish(event: DomainEvent) -> None`
- `subscribe(event_type: type[DomainEvent], handler: Callable) -> None`
- `unsubscribe(event_type: type[DomainEvent], handler: Callable) -> None`
- `publish_batch(events: list[DomainEvent]) -> None`

#### `IStorageBackend` (abstract)
Port for reading and writing artifacts (models, datasets, reports).
- `write(path: str, data: bytes) -> str`
- `read(path: str) -> bytes`
- `exists(path: str) -> bool`
- `delete(path: str) -> None`
- `list_prefix(prefix: str) -> list[str]`

#### `IInferenceEngine` (abstract)
Port for running model inference and batch evaluation.
- `predict(model_id: str, version: str, inputs: Any) -> Any`
- `predict_proba(model_id: str, version: str, inputs: Any) -> Any`
- `batch_predict(model_id: str, version: str, dataset: DataArtifactPointer) -> Any`
- `get_internal_representations(model_id: str, version: str, inputs: Any) -> Any`

#### `IExplainabilityEngine` (abstract)
Port for generating local and global explanations.
- `explain_local(model_id: str, instance: Any) -> dict`
- `explain_global(model_id: str, dataset: DataArtifactPointer) -> dict`
- `explain_interactions(model_id: str, dataset: DataArtifactPointer) -> dict`
- `explain_batch(model_id: str, instances: list) -> list[dict]`

#### `INotifier` (abstract)
Port for sending alerts and notifications.
- `send_alert(title: str, body: str, severity: Severity, metadata: dict) -> None`
- `send_report(report: Report, recipients: list[str]) -> None`
- `send_incident_update(incident_id: str, status: str, details: str) -> None`

#### `ICapabilityRegistry` (abstract)
Port for checking which features are enabled.
- `is_enabled(capability: str) -> bool`
- `get_tier() -> CapabilityTier`
- `list_enabled() -> list[str]`
- `require(capability: str) -> None`  *(raises CapabilityNotAvailableError if not enabled)*

---

## 3. `holmes.events`

Domain events and the default in-process event bus.

### Classes

#### `DomainEvent` (abstract base class)
Base for all domain events.
- **Attributes:** `event_id: UUID`, `occurred_at: datetime`, `source: str`, `correlation_id: UUID | None`
- **Methods:** `event_type() -> str`, `to_dict() -> dict`

#### `DatasetRegistered(DomainEvent)`
Fired when a new dataset is registered in MEMS.
- **Attributes:** `dataset_id: str`, `dataset_name: str`, `schema_id: str`

#### `ModelRegistered(DomainEvent)`
Fired when a new model version is registered in MEMS.
- **Attributes:** `model_id: str`, `version: str`, `framework: str`

#### `PredictionCompleted(DomainEvent)`
Fired after each production prediction batch or single inference.
- **Attributes:** `model_id: str`, `version: str`, `prediction_id: str`, `input_hash: str`, `output_summary: dict`, `latency_ms: float`

#### `DriftDetected(DomainEvent)`
Fired by Drift Monitoring when a threshold is crossed.
- **Attributes:** `model_id: str`, `drift_type: str`, `severity: Severity`, `affected_features: list[str]`, `drift_score: float`

#### `FindingPublished(DomainEvent)`
Fired when a subsystem emits a new finding.
- **Attributes:** `finding_id: UUID`, `finding_type: FindingType`, `engine: str`, `severity: Severity`

#### `InvestigationStarted(DomainEvent)`
Fired when an investigation run begins.
- **Attributes:** `run_id: UUID`, `engine: str`, `triggered_by: str`, `context: dict`

#### `InvestigationCompleted(DomainEvent)`
Fired when an investigation run finishes.
- **Attributes:** `run_id: UUID`, `engine: str`, `finding_count: int`, `health_score: float`, `duration_ms: int`

#### `IncidentOpened(DomainEvent)`
Fired when a new production incident is created.
- **Attributes:** `incident_id: UUID`, `model_id: str`, `root_cause_summary: str`, `severity: Severity`

#### `IncidentResolved(DomainEvent)`
Fired when an incident is closed.
- **Attributes:** `incident_id: UUID`, `resolution_summary: str`, `time_to_resolve_hours: float`

#### `InProcessEventBus`
Default synchronous event bus. No external infrastructure. Used by OSS tier.
- **Attributes:** `_subscribers: dict[str, list[Callable]]`
- `publish(event: DomainEvent) -> None`
- `subscribe(event_type: type[DomainEvent], handler: Callable) -> None`
- `unsubscribe(event_type: type[DomainEvent], handler: Callable) -> None`

#### `EventHandler` (abstract base class)
Base class for all event handlers.
- `handle(event: DomainEvent) -> None` *(abstract)*

---

## 4. `holmes.application`

Application layer: commands, handlers, orchestration, and the capability registry.

### Classes

#### `CapabilityRegistry`
Central feature-flag system. Implements `ICapabilityRegistry`.
- **Attributes:** `_tier: CapabilityTier`, `_enabled: set[str]`, `_all_capabilities: dict[str, CapabilityTier]`
- `is_enabled(capability: str) -> bool`
- `register(capability: str, tier: CapabilityTier) -> None`
- `load_from_license(license_key: str) -> None`
- `require(capability: str) -> None`

#### `InvestigationOrchestrator`
Coordinates the three investigation engines. Routes commands to the right engine.
- **Dependencies:** `DataInvestigationEngine`, `ModelInvestigationEngine`, `SystemInvestigationEngine`, `IEventBus`
- `run_data_investigation(cmd: InvestigateDatasetCommand) -> DataHealthReport`
- `run_model_investigation(cmd: InvestigateModelCommand) -> ModelHealthReport`
- `start_system_monitoring(cmd: StartDriftMonitoringCommand) -> None`
- `stop_system_monitoring(model_id: str) -> None`

#### `HolmesConfig`
Top-level platform configuration loaded at startup.
- **Attributes:** `tier: CapabilityTier`, `storage_backend: str`, `event_bus_backend: str`, `model_store_backend: str`, `data_catalog_backend: str`, `inference_backend: str`, `explanation_backend: str`
- **Methods:** `from_env() -> HolmesConfig`, `from_file(path: str) -> HolmesConfig`, `validate() -> None`

#### Commands (dataclasses / value objects)

| Class | Key Fields |
|---|---|
| `InvestigateDatasetCommand` | `dataset_id`, `subsystems: list[str] \| None`, `config: dict` |
| `InvestigateModelCommand` | `model_id`, `version`, `dataset_id`, `subsystems: list[str] \| None` |
| `StartDriftMonitoringCommand` | `model_id`, `version`, `thresholds: dict`, `alert_channels: list[str]` |
| `StopDriftMonitoringCommand` | `model_id` |
| `RunRootCauseAnalysisCommand` | `model_id`, `time_window_start`, `time_window_end`, `trigger_finding_id` |
| `GetReportCommand` | `report_id: UUID` |
| `GetFindingsCommand` | `filters: FindingQuery` |

#### Handlers

| Class | Handles | Returns |
|---|---|---|
| `InvestigateDatasetHandler` | `InvestigateDatasetCommand` | `DataHealthReport` |
| `InvestigateModelHandler` | `InvestigateModelCommand` | `ModelHealthReport` |
| `DriftMonitoringHandler` | `StartDriftMonitoringCommand` | `None` |
| `RootCauseAnalysisHandler` | `RunRootCauseAnalysisCommand` | `RootCauseReport` |

---

## 5. `holmes.mems`

Model Entity Metadata Store — the central registry for models, datasets, lineage, and baselines.

### Classes

#### `ModelRegistry`
CRUD interface for model registrations and versions.
- `register(model: ModelVersion) -> str`
- `get(model_id: str, version: str) -> ModelVersion`
- `list(filters: dict | None) -> list[ModelVersion]`
- `get_latest(model_id: str) -> ModelVersion`
- `update_health_status(model_id: str, version: str, score: HealthScore) -> None`
- `get_deployment_history(model_id: str) -> list[DeploymentRecord]`

#### `DatasetRegistry`
CRUD interface for dataset registrations.
- `register(binding: DatasetBinding) -> str`
- `get(dataset_id: str) -> DatasetBinding`
- `update_health_status(dataset_id: str, score: HealthScore) -> None`
- `list(filters: dict | None) -> list[DatasetBinding]`

#### `ModelVersion`
A versioned model artifact with full metadata.
- **Attributes:** `model_id: str`, `version: str`, `framework: str`, `artifact_pointer: DataArtifactPointer`, `training_run_id: str`, `feature_names: list[str]`, `target_name: str`, `performance_metrics: dict`, `created_at: datetime`, `tags: dict`

#### `DatasetBinding`
A dataset with its splits, schema, and statistical profile.
- **Attributes:** `dataset_id: str`, `name: str`, `schema: DataSchema`, `profile: DataProfile`, `splits: dict[str, DataArtifactPointer]`, `temporal_column: str | None`, `label_column: str`, `task_type: TaskType`

#### `DataProfile`
Statistical summary of a dataset (computed once, referenced everywhere).
- **Attributes:** `feature_stats: dict[str, FeatureStats]`, `row_count: int`, `label_distribution: dict`, `missing_rates: dict[str, float]`, `computed_at: datetime`

#### `FeatureStats`
Per-feature statistics stored in a DataProfile.
- **Attributes:** `name: str`, `dtype: str`, `mean: float | None`, `std: float | None`, `min: float | None`, `max: float | None`, `quantiles: dict`, `cardinality: int | None`, `null_rate: float`

#### `DataSchema`
Schema definition with type constraints and validation rules.
- **Attributes:** `columns: list[ColumnDef]`, `version: str`, `created_at: datetime`
- **Methods:** `validate(df) -> list[str]`, `diff(other: DataSchema) -> list[str]`

#### `ColumnDef`
Definition of a single column within a schema.
- **Attributes:** `name: str`, `dtype: str`, `nullable: bool`, `min_value: Any | None`, `max_value: Any | None`, `allowed_values: list | None`

#### `TaskType` (enum)
- Values: `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `REGRESSION`, `RANKING`, `ANOMALY_DETECTION`

#### `TrainingRun`
Records a single model training run with parameters and metrics.
- **Attributes:** `run_id: str`, `model_id: str`, `params: dict`, `metrics: dict`, `dataset_id: str`, `started_at: datetime`, `finished_at: datetime`

#### `DeploymentRecord`
Tracks a model deployment to production.
- **Attributes:** `deployment_id: str`, `model_id: str`, `version: str`, `environment: str`, `deployed_at: datetime`, `rolled_back_at: datetime | None`, `traffic_percent: float`

#### `ModelLineage`
Tracks the full provenance of a model including data and pipeline dependencies.
- **Attributes:** `model_id: str`, `version: str`, `parent_dataset_ids: list[str]`, `upstream_model_ids: list[str]`, `pipeline_steps: list[PipelineStep]`

#### `PipelineStep`
A single step in a data or ML pipeline.
- **Attributes:** `name: str`, `type: str`, `params: dict`, `input_ids: list[str]`, `output_ids: list[str]`

---

## 6. `holmes.iss`

Inference & Simulation Service — runs model predictions, batch evaluations, and counterfactual simulations.

### Classes

#### `InferenceService`
Main entry point. Coordinates all ISS operations.
- **Dependencies:** `IInferenceEngine`, `IModelStore`, `ModelEvaluator`
- `predict(model_id: str, version: str, inputs: Any) -> Any`
- `evaluate(model_id: str, version: str, dataset: DataArtifactPointer, labels: Any) -> dict`
- `simulate_counterfactual(model_id: str, scenario: CounterfactualScenario) -> CounterfactualResult`

#### `BatchInferenceRunner`
Efficient batch prediction over large datasets.
- `run(model_id: str, version: str, dataset: DataArtifactPointer, batch_size: int) -> Any`
- `run_with_perturbation(model_id: str, version: str, dataset: DataArtifactPointer, perturber: PerturbationEngine) -> list[Any]`

#### `CounterfactualSimulator`
Answers "what-if" questions by simulating alternative scenarios.
- `simulate_feature_removal(model_id: str, feature: str, dataset: DataArtifactPointer) -> CounterfactualResult`
- `simulate_distribution_shift(model_id: str, shift: dict, dataset: DataArtifactPointer) -> CounterfactualResult`
- `simulate_retraining(model_id: str, new_dataset_id: str) -> CounterfactualResult`

#### `CounterfactualScenario`
Describes the parameters of a counterfactual simulation.
- **Attributes:** `scenario_type: str`, `params: dict`, `baseline_model_id: str`, `baseline_version: str`

#### `CounterfactualResult`
Output of a counterfactual simulation.
- **Attributes:** `scenario: CounterfactualScenario`, `baseline_metrics: dict`, `counterfactual_metrics: dict`, `delta: dict`, `confidence: float`

#### `PerturbationEngine`
Applies configurable perturbations to input features for stability and robustness tests.
- `add_gaussian_noise(feature: str, sigma: float) -> PerturbationEngine`
- `add_uniform_noise(feature: str, scale: float) -> PerturbationEngine`
- `mask_feature(feature: str) -> PerturbationEngine`
- `apply(df) -> Any`

#### `ModelEvaluator`
Computes metrics given predictions and ground truth labels.
- `compute_metrics(y_true: Any, y_pred: Any, y_proba: Any | None, task_type: TaskType) -> dict`
- `compute_calibration(y_true: Any, y_proba: Any, n_bins: int) -> dict`
- `compute_per_slice(y_true: Any, y_pred: Any, slices: dict) -> dict[str, dict]`

#### `StreamingInferenceConsumer`
Subscribes to the event bus and processes `PredictionCompleted` events in real time.
- `start(model_id: str) -> None`
- `stop() -> None`
- `on_prediction(event: PredictionCompleted) -> None`

---

## 7. `holmes.eos`

Explanation & Observability Service — generates local and global explanations and detects explanation drift.

### Classes

#### `ExplanationService`
Central entry point for all explanation requests.
- **Dependencies:** `IExplainabilityEngine`, `ExplanationCache`
- `explain_instance(model_id: str, version: str, instance: Any) -> LocalExplanation`
- `explain_dataset(model_id: str, version: str, dataset: DataArtifactPointer) -> GlobalExplanation`
- `explain_interactions(model_id: str, version: str, dataset: DataArtifactPointer) -> InteractionExplanation`
- `compare_explanations(explanation_a: GlobalExplanation, explanation_b: GlobalExplanation) -> ExplanationDiff`

#### `LocalExplanation`
Explanation for a single instance.
- **Attributes:** `instance_id: str`, `feature_contributions: dict[str, float]`, `baseline_value: float`, `prediction: Any`, `method: str`

#### `GlobalExplanation`
Aggregated explanation across a dataset.
- **Attributes:** `feature_importances: dict[str, float]`, `feature_rank: list[str]`, `method: str`, `sample_count: int`

#### `InteractionExplanation`
Pairwise and higher-order feature interaction values.
- **Attributes:** `interaction_matrix: dict[tuple, float]`, `top_interactions: list[tuple]`, `synergistic_pairs: list[tuple]`, `redundant_pairs: list[tuple]`

#### `ExplanationDiff`
Comparison between two explanations (e.g. before/after drift).
- **Attributes:** `rank_change: dict[str, int]`, `importance_delta: dict[str, float]`, `stability_score: float`

#### `ExplanationCache`
Caches expensive explanation computations to avoid recomputation.
- `get(cache_key: str) -> Any | None`
- `set(cache_key: str, value: Any, ttl_seconds: int) -> None`
- `invalidate(model_id: str) -> None`

#### `ExplanationDriftDetector`
Detects when explanation patterns shift significantly over time.
- `check(baseline: GlobalExplanation, current: GlobalExplanation) -> Finding | None`
- `is_stable(baseline: GlobalExplanation, current: GlobalExplanation, threshold: float) -> bool`

---

## 8. `holmes.edas`

Evidence & Diagnostics Artifact Store — persists and queries all findings, reports, and health scores.

### Classes

#### `FindingRepository`
Persists and queries `Finding` objects.
- **Dependencies:** `IStorageBackend`
- `save(finding: Finding) -> None`
- `save_batch(findings: list[Finding]) -> None`
- `get(finding_id: UUID) -> Finding`
- `query(q: FindingQuery) -> list[Finding]`
- `count(q: FindingQuery) -> int`

#### `ReportRepository`
Persists and queries health reports.
- `save(report: Report) -> None`
- `get(report_id: UUID) -> Report`
- `list_for_model(model_id: str, limit: int) -> list[Report]`
- `list_for_dataset(dataset_id: str, limit: int) -> list[Report]`
- `get_latest_for_model(model_id: str) -> Report | None`

#### `IncidentRepository`
Manages the full lifecycle of production incidents.
- `open(incident: Incident) -> str`
- `update(incident_id: str, updates: dict) -> None`
- `resolve(incident_id: str, resolution: str) -> None`
- `get(incident_id: str) -> Incident`
- `list_open(model_id: str | None) -> list[Incident]`
- `list_resolved(model_id: str, days: int) -> list[Incident]`

#### `Incident`
A production incident with full lifecycle tracking.
- **Attributes:** `incident_id: UUID`, `model_id: str`, `status: IncidentStatus`, `severity: Severity`, `root_cause_findings: list[UUID]`, `action_plan_id: UUID | None`, `opened_at: datetime`, `resolved_at: datetime | None`

#### `IncidentStatus` (enum)
- Values: `DETECTED`, `INVESTIGATING`, `MITIGATING`, `RESOLVED`, `CLOSED`

#### `HealthScoreRepository`
Time-series store for health scores.
- `record(model_id: str, score: HealthScore, timestamp: datetime) -> None`
- `get_history(model_id: str, days: int) -> list[tuple[datetime, HealthScore]]`
- `get_trend(model_id: str, days: int) -> float`

#### `FindingQuery`
Builder for composing complex finding queries.
- `for_model(model_id: str) -> FindingQuery`
- `for_dataset(dataset_id: str) -> FindingQuery`
- `with_type(finding_type: FindingType) -> FindingQuery`
- `with_severity(min_severity: Severity) -> FindingQuery`
- `since(dt: datetime) -> FindingQuery`
- `limit(n: int) -> FindingQuery`
- `build() -> dict`

#### `FindingAggregator`
Groups and deduplicates findings from multiple subsystems before storing.
- `deduplicate(findings: list[Finding]) -> list[Finding]`
- `group_by_segment(findings: list[Finding]) -> dict[str, list[Finding]]`
- `merge_related(findings: list[Finding]) -> list[Finding]`

---

## 9. `holmes.data_investigation`

Data Investigation Engine — coordinates 6 subsystems to produce a `DataHealthReport`.

### Classes

#### `DataInvestigationEngine`
Top-level coordinator. Resolves which subsystems to run based on config and tier.
- **Dependencies:** `CapabilityRegistry`, `IEventBus`, `FindingRepository`, all 6 subsystems
- `run(ctx: InvestigationContext) -> DataHealthReport`
- `run_subsystem(name: str, ctx: InvestigationContext) -> list[Finding]`

---

### 9.1 Data Profiling & Integrity Subsystem

#### `DataProfileInvestigator`
Coordinates all profiling analyses and merges their findings.
- `investigate(dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `MissingValueAnalyzer`
Detects missing value patterns and tests MCAR / MAR / MNAR mechanisms.
- `analyze(df) -> list[Finding]`
- `test_mcar(df) -> dict`
- `test_mar(df) -> dict`
- `find_missing_clusters(df) -> list[dict]`

#### `DuplicateDetector`
Finds exact and near-duplicate records and assesses label conflicts.
- `detect_exact(df) -> list[Finding]`
- `detect_near(df, threshold: float) -> list[Finding]`
- `check_label_conflicts(df, label_col: str) -> list[Finding]`

#### `OutlierDiscovery`
Multivariate outlier detection and classification.
- `detect(df) -> list[Finding]`
- `classify_outlier(row: dict, profile: DataProfile) -> str`  *→ "valid_extreme" | "potential_error" | "distributional_anomaly"*

#### `SchemaValidator`
Validates actual data against the registered `DataSchema`.
- `validate(df, schema: DataSchema) -> list[Finding]`
- `detect_drift(old_schema: DataSchema, new_schema: DataSchema) -> list[Finding]`

#### `DataFreshnessChecker`
Detects stale records, late-arriving data, and temporal gaps.
- `check(df, timestamp_col: str) -> list[Finding]`
- `detect_gaps(df, timestamp_col: str, expected_freq: str) -> list[Finding]`

---

### 9.2 Leakage Detection Subsystem

#### `LeakageDetectionSubsystem`
Coordinates all leakage analyses.
- `run(dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `TemporalLeakageDetector`
Checks for features that use future information at prediction time.
- `check(df, temporal_col: str, features: list[str], event_col: str) -> list[Finding]`
- `simulate_point_in_time(df, temporal_col: str) -> list[Finding]`

#### `PreprocessingLeakageAuditor`
Detects when preprocessing used test/validation data to fit transformers.
- `audit(train_df, test_df, pipeline) -> list[Finding]`
- `compare_transform_params(subset_a, subset_b, pipeline) -> list[Finding]`

#### `ProxyVariableIdentifier`
Identifies features with spurious predictive power due to data collection artifacts.
- `identify(df, label_col: str, features: list[str]) -> list[Finding]`
- `compute_predictive_power_stability(df, feature: str, label_col: str) -> float`

#### `TrainTestOverlapChecker`
Detects entity-level overlap between training and test splits.
- `check(train_df, test_df, entity_col: str) -> list[Finding]`
- `compute_overlap_index(train_df, test_df, entity_col: str) -> float`

#### `LeakageScorer`
Estimates how much a leaky feature inflates validation metrics.
- `score(model_id: str, suspect_feature: str, dataset: DatasetBinding) -> float`

---

### 9.3 Label Quality Subsystem

#### `LabelQualitySubsystem`
Coordinates label quality analyses.
- `run(dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `MislabelDetector`
Uses confident learning to surface likely mislabeled instances.
- `detect(df, label_col: str, model_proba) -> list[Finding]`
- `compute_noise_matrix(df, label_col: str, model_proba) -> dict`

#### `AnnotatorAgreementAnalyzer`
Computes inter-annotator agreement for multi-source labeled data.
- `analyze(df, label_cols: list[str]) -> list[Finding]`
- `compute_cohens_kappa(labels_a, labels_b) -> float`
- `find_disagreement_segments(df, label_cols: list[str]) -> list[dict]`

#### `LabelDriftDetector`
Detects shifts in label distributions across time or data batches.
- `detect(df_old, df_new, label_col: str) -> list[Finding]`
- `detect_definition_change(df, label_col: str, timestamp_col: str) -> list[Finding]`

#### `NoiseMatrixEstimator`
Estimates the label noise transition matrix per segment.
- `estimate(df, label_col: str, model_proba) -> dict`
- `estimate_per_segment(df, label_col: str, segment_col: str, model_proba) -> dict`

#### `RelabelPrioritizer`
Ranks instances for re-labeling by expected performance impact.
- `prioritize(df, mislabel_scores: dict) -> list[dict]`
- `estimate_relabel_benefit(df, label_col: str, noise_matrix: dict) -> float`

---

### 9.4 Coverage Analysis Subsystem

#### `CoverageAnalysisSubsystem`
Coordinates coverage analyses and produces a `CoverageMap`.
- `run(dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `DensityEstimator`
Builds a density model of the training data distribution.
- `fit(df) -> None`
- `score_samples(df) -> list[float]`
- `find_low_density_regions(df, threshold: float) -> list[dict]`

#### `GapDetector`
Compares training density against a reference distribution to find gaps.
- `detect(train_df, reference_df) -> list[Finding]`
- `find_intersectional_gaps(train_df, reference_df, feature_pairs: list[tuple]) -> list[Finding]`

#### `ExtrapolationRiskScorer`
Assigns extrapolation risk to new instances outside the training hull.
- `fit(train_df) -> None`
- `score(instance: dict) -> float`
- `flag_high_risk(df, threshold: float) -> list[Finding]`

#### `CoverageMap`
Data structure for the full coverage map (used by the frontend/DIL).
- **Attributes:** `safe_zones: list[dict]`, `gap_zones: list[dict]`, `extrapolation_zones: list[dict]`, `feature_coverage_scores: dict[str, float]`

#### `BusinessCriticalityWeighter`
Weights coverage gaps by business impact to prioritize what matters.
- `weight(gaps: list[dict], business_metrics: dict) -> list[dict]`

---

### 9.5 Signal & Predictability Subsystem

#### `SignalPredictabilitySubsystem`
Coordinates signal analyses and produces a `SignalReport`.
- `run(dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `MutualInformationEstimator`
Estimates pairwise and conditional mutual information between features and target.
- `estimate(df, feature: str, label_col: str) -> float`
- `estimate_all(df, label_col: str) -> dict[str, float]`

#### `RedundancyAnalyzer`
Finds groups of features with overlapping predictive information.
- `analyze(df, label_col: str) -> list[Finding]`
- `find_redundant_pairs(df, label_col: str, threshold: float) -> list[tuple]`
- `find_synergistic_pairs(df, label_col: str) -> list[tuple]`

#### `BayesErrorEstimator`
Estimates the irreducible error bound for the given dataset.
- `estimate(df, label_col: str, method: str) -> float`
- `compute_performance_ceiling(df, label_col: str, task_type: TaskType) -> dict`

#### `LearningCurveFitter`
Fits power-law learning curves to project performance vs data volume.
- `fit(subset_sizes: list[int], metrics: list[float]) -> None`
- `project(target_size: int) -> float`
- `is_data_limited(current_size: int, saturation_threshold: float) -> bool`

#### `FeatureValueAssessor`
Estimates the potential performance gain from improving a specific feature.
- `assess(feature: str, df, label_col: str) -> float`
- `assess_all(df, label_col: str) -> dict[str, float]`

---

### 9.6 Data Report Aggregation Subsystem

#### `DataReportAggregationSubsystem`
Compiles all findings from the 5 subsystems into the final `DataHealthReport`.
- `run(all_findings: list[Finding], ctx: InvestigationContext) -> DataHealthReport`

#### `DataHealthReport(Report)`
The final output of the Data Investigation Engine.
- **Attributes (extends Report):** `coverage_map: CoverageMap | None`, `label_noise_estimate: float | None`, `leakage_risk_features: list[str]`

#### `DataHealthScoreCalculator`
Computes the 0–100 data health score from subsystem findings.
- `calculate(findings: list[Finding]) -> HealthScore`
- `compute_dimension_scores(findings: list[Finding]) -> dict[str, float]`

#### `FindingDeduplicator`
Merges findings that describe the same underlying issue.
- `deduplicate(findings: list[Finding]) -> list[Finding]`
- `find_related_groups(findings: list[Finding]) -> list[list[Finding]]`

#### `SeverityNormalizer`
Normalizes severity and confidence scores across subsystems.
- `normalize(findings: list[Finding]) -> list[Finding]`

---

## 10. `holmes.model_investigation`

Model Investigation Engine — coordinates 7 subsystems to produce a `ModelHealthReport`.

### Classes

#### `ModelInvestigationEngine`
Top-level coordinator. Resolves subsystems based on config and tier.
- **Dependencies:** `CapabilityRegistry`, `IEventBus`, `FindingRepository`, all 7 subsystems
- `run(ctx: InvestigationContext) -> ModelHealthReport`

---

### 10.1 Generalization & Robustness Subsystem

#### `RobustnessSubsystem`
Produces a `RobustnessReport` covering shift tolerance and adversarial vulnerability.
- `run(model: ModelVersion, dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `ShiftSimulator`
Simulates covariate shift, label shift, and concept drift at varying magnitudes.
- `simulate_covariate_shift(df, magnitude: float) -> Any`
- `simulate_label_shift(df, shift_factor: float) -> Any`
- `generate_ood_samples(df, n_samples: int) -> Any`

#### `RobustnessCurveBuilder`
Measures and records performance degradation vs shift magnitude.
- `build(model_id: str, version: str, df, shift_range: list[float]) -> dict`

#### `AdversarialAttacker`
Runs adversarial attacks to find small perturbations causing large prediction changes.
- `fgsm_attack(model_id: str, version: str, inputs: Any, epsilon: float) -> Any`
- `query_attack(model_id: str, version: str, inputs: Any, budget: int) -> Any`
- `compute_vulnerability_map(model_id: str, version: str, df) -> list[Finding]`

#### `OODDetector`
Trains an OOD detector using Mahalanobis distance in the model's feature space.
- `fit(model_id: str, version: str, train_df) -> None`
- `score(instance: Any) -> float`
- `define_boundary(percentile: float) -> float`

---

### 10.2 Failure Region Discovery Subsystem

#### `FailureRegionSubsystem`
Discovers subpopulations where the model fails significantly worse than average.
- `run(model: ModelVersion, dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `SliceFinder`
Searches all feature conditions to find high-error slices.
- `find(df, errors: list[float], min_slice_size: int) -> list[Slice]`
- `rank_by_error_rate(slices: list[Slice]) -> list[Slice]`

#### `Slice`
Represents a discovered failure region.
- **Attributes:** `conditions: dict`, `size: int`, `error_rate: float`, `baseline_error_rate: float`, `impact_score: float`

#### `SliceCharacterizer`
Computes rich statistics for each discovered slice.
- `characterize(s: Slice, df) -> dict`
- `is_systematic(s: Slice, model_ids: list[str]) -> bool`

#### `SliceAttributionAnalyzer`
Computes feature attributions specifically for instances within a failure slice.
- `analyze(s: Slice, df, explanation_service: ExplanationService) -> dict`
- `compare_to_global(slice_importance: dict, global_importance: dict) -> dict`

#### `EmergentFailureMonitor`
Tracks slice error rates over time and alerts when new failures emerge.
- `register_slice(s: Slice) -> None`
- `update(model_id: str, new_df) -> list[Finding]`

#### `FairnessSliceIdentifier`
Flags slices defined by protected attributes as potential bias indicators.
- `identify(slices: list[Slice], protected_attributes: list[str]) -> list[Finding]`

---

### 10.3 Calibration & Confidence Subsystem

#### `CalibrationSubsystem`
Evaluates probability calibration globally and per segment.
- `run(model: ModelVersion, dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `CalibrationMetricsCalculator`
Computes ECE, MCE, and Brier score with confidence intervals.
- `compute_ece(y_true, y_proba, n_bins: int) -> float`
- `compute_mce(y_true, y_proba, n_bins: int) -> float`
- `compute_brier(y_true, y_proba) -> float`

#### `ReliabilityDiagramBuilder`
Builds the data structure for a reliability (calibration) diagram.
- `build(y_true, y_proba, n_bins: int) -> dict`

#### `SubpopulationCalibrationAnalyzer`
Identifies segments where calibration is significantly worse.
- `analyze(df, y_true, y_proba, segment_cols: list[str]) -> list[Finding]`

#### `ThresholdOptimizer`
Finds the optimal decision threshold given a business utility function.
- `optimize(y_true, y_proba, cost_fn_false_pos: float, cost_fn_false_neg: float) -> float`
- `optimize_per_segment(df, y_true, y_proba, segment_col: str, cost_matrix: dict) -> dict`

#### `CalibrationDriftMonitor`
Tracks calibration over time in production.
- `update(model_id: str, window_y_true, window_y_proba) -> list[Finding]`

---

### 10.4 Shortcut Learning Detection Subsystem `[PRO]`

#### `ShortcutDetectionSubsystem`
Identifies features the model exploits as spurious correlations.
- **Tier:** `CapabilityTier.PRO`
- `run(model: ModelVersion, dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `SpuriousFeatureFinder`
Measures importance stability across environments to flag shortcuts.
- `find(model_id: str, version: str, environments: list[Any]) -> list[Finding]`
- `compute_importance_stability(importances_by_env: list[dict]) -> dict[str, float]`

#### `CounterfactualFeatureRemover`
Tests model performance after removing or randomizing a suspect feature.
- `test_removal(model_id: str, version: str, feature: str, dataset: DatasetBinding) -> CounterfactualResult`

#### `DomainKnowledgeInjector`
Allows user-specified suspect or protected features to be tested.
- `inject(suspect_features: list[str], protected_features: list[str]) -> None`
- `get_suspect_features() -> list[str]`

#### `ShortcutStrengthScorer`
Computes a reliance score per feature based on stability and counterfactual tests.
- `score(feature: str, stability: float, counterfactual_result: CounterfactualResult) -> float`

---

### 10.5 Prediction Stability Subsystem

#### `PredictionStabilitySubsystem`
Measures prediction consistency under perturbations and across model seeds.
- `run(model: ModelVersion, dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `LocalPerturbationTester`
Applies small input noise and measures prediction variance.
- `test(model_id: str, version: str, df, n_perturbations: int) -> list[Finding]`
- `compute_fragility_score(model_id: str, version: str, instance: Any) -> float`

#### `DecisionBoundaryEstimator`
Estimates the distance of each instance to the model's decision boundary.
- `estimate(model_id: str, version: str, df) -> list[float]`
- `flag_boundary_proximity(distances: list[float], threshold: float) -> list[Finding]`

#### `TrainingVarianceAssessor`
Compares predictions across models trained with different random seeds.
- `assess(model_ids: list[str], version: str, df) -> list[Finding]`
- `compute_seed_disagreement(predictions: list[Any]) -> float`

#### `ExplanationStabilityChecker`
Checks whether feature importance rankings are consistent across perturbation runs.
- `check(model_id: str, version: str, df, n_runs: int) -> list[Finding]`
- `compute_rank_stability(explanations: list[dict]) -> float`

#### `FragilitySegmentDetector`
Aggregates fragility scores across subpopulations.
- `detect(fragility_scores: dict, df) -> list[Finding]`

---

### 10.6 Feature Dependency & Interaction Subsystem `[PRO]`

#### `FeatureInteractionSubsystem`
Uncovers complex feature relationships and interactions.
- **Tier:** `CapabilityTier.PRO`
- `run(model: ModelVersion, dataset: DatasetBinding, ctx: InvestigationContext) -> list[Finding]`

#### `InteractionGraphBuilder`
Computes pairwise and higher-order Shapley interaction values.
- `build(model_id: str, version: str, df) -> InteractionGraph`

#### `InteractionGraph`
Graph representation of feature interactions.
- **Attributes:** `nodes: list[str]`, `edges: dict[tuple, float]`, `synergistic: list[tuple]`, `redundant: list[tuple]`, `antagonistic: list[tuple]`

#### `SuppressionDetector`
Finds features that suppress the importance of other features.
- `detect(interaction_graph: InteractionGraph) -> list[Finding]`

#### `FeatureClusterer`
Groups features into functional modules based on interaction patterns.
- `cluster(interaction_graph: InteractionGraph) -> dict[str, list[str]]`

#### `MediationAnalyzer`
Determines whether a feature's influence is direct or mediated through others.
- `analyze(feature: str, interaction_graph: InteractionGraph, causal_order: list[str] | None) -> dict`

#### `ContextDependentImportanceCalculator`
Shows how a feature's importance varies as a function of another feature's value.
- `calculate(feature_a: str, feature_b: str, model_id: str, version: str, df) -> dict`

---

### 10.7 Model Report Aggregation Subsystem

#### `ModelReportAggregationSubsystem`
Compiles all findings into the final `ModelHealthReport`.
- `run(all_findings: list[Finding], ctx: InvestigationContext) -> ModelHealthReport`

#### `ModelHealthReport(Report)`
Final output of the Model Investigation Engine.
- **Attributes (extends Report):** `robustness_summary: dict | None`, `failure_slices: list[Slice]`, `deployment_readiness: DeploymentReadiness`

#### `DeploymentReadiness`
Go/no-go deployment recommendation based on findings.
- **Attributes:** `is_ready: bool`, `blocking_findings: list[Finding]`, `warnings: list[Finding]`, `score: float`

#### `ModelHealthScoreCalculator`
Computes the 0–100 model health score.
- `calculate(findings: list[Finding]) -> HealthScore`

#### `CrossFindingSynthesizer`
Builds coherent narratives by combining related findings across subsystems.
- `synthesize(findings: list[Finding]) -> list[Finding]`

---

## 11. `holmes.system_investigation`

System Investigation Engine — monitors production, detects drift, performs root cause analysis, and manages incidents.

### Classes

#### `SystemInvestigationEngine`
Top-level coordinator for all 5 system investigation subsystems.
- **Dependencies:** `CapabilityRegistry`, `IEventBus`, `FindingRepository`, `StreamingInferenceConsumer`, all 5 subsystems
- `start_monitoring(model_id: str, config: dict) -> None`
- `stop_monitoring(model_id: str) -> None`
- `run_root_cause_analysis(ctx: InvestigationContext) -> RootCauseReport`

---

### 11.1 Production Drift Monitoring Subsystem

#### `DriftMonitoringSubsystem`
Continuously monitors production data and prediction streams for drift.
- `start(model_id: str, version: str, baseline_profile: DataProfile) -> None`
- `on_prediction_batch(event: PredictionCompleted, df) -> list[Finding]`

#### `DataDriftDetector`
Compares production feature distributions against training baselines.
- `detect(production_df, baseline_profile: DataProfile) -> list[Finding]`
- `ks_test(production_series, baseline_stats: dict) -> float`
- `js_distance(production_series, baseline_stats: dict) -> float`
- `detect_multivariate(production_df, baseline_profile: DataProfile) -> Finding | None`

#### `PredictionDriftDetector`
Tracks the distribution of predictions over time.
- `detect(production_predictions, baseline_prediction_dist: dict) -> list[Finding]`
- `detect_confidence_shift(production_probas, baseline_proba_dist: dict) -> Finding | None`

#### `PerformanceMonitor`
Computes performance metrics when delayed ground truth labels become available.
- `update(y_true, y_pred, y_proba, timestamp: datetime) -> list[Finding]`
- `detect_degradation(current_metrics: dict, baseline_metrics: dict) -> list[Finding]`

#### `AnomalyBasedDriftTrigger`
Uses unsupervised anomaly detection to identify unusual production periods.
- `fit(baseline_representations) -> None`
- `score(production_representations) -> float`
- `trigger_if_anomalous(score: float, threshold: float) -> Finding | None`

#### `DriftThresholdManager`
Manages custom drift thresholds per feature and segment.
- `set_threshold(feature: str, threshold: float) -> None`
- `set_global_threshold(threshold: float) -> None`
- `get_threshold(feature: str) -> float`

#### `DriftAlertPublisher`
Publishes `DriftDetected` events when thresholds are exceeded.
- `publish_if_exceeded(findings: list[Finding], bus: IEventBus) -> None`

---

### 11.2 Root Cause Analysis Subsystem `[PRO]`

#### `RootCauseAnalysisSubsystem`
Systematically investigates and ranks the most likely causes of a performance issue.
- **Tier:** `CapabilityTier.PRO`
- `run(ctx: InvestigationContext) -> RootCauseReport`

#### `HypothesisGenerator`
Automatically generates a list of plausible root cause hypotheses.
- `generate(drift_findings: list[Finding], model: ModelVersion) -> list[Hypothesis]`

#### `Hypothesis`
A candidate root cause with associated evidence and confidence.
- **Attributes:** `type: str`, `description: str`, `confidence: float`, `supporting_findings: list[UUID]`

#### `PerformanceAttributionDecomposer`
Decomposes a total performance drop into per-feature and per-segment contributions.
- `decompose(production_df, baseline_df, model_id: str, version: str) -> dict`

#### `CounterfactualValidator`
Validates hypotheses by running counterfactual simulations.
- `validate(hypothesis: Hypothesis, simulator: CounterfactualSimulator) -> float`

#### `TimelineReconstructor`
Correlates performance degradation onset with external events.
- `reconstruct(model_id: str, events: list[dict], performance_history: list[tuple]) -> list[dict]`

#### `PipelineBlameDecomposer`
Identifies which pipeline stage introduced the performance error.
- `decompose(lineage: ModelLineage, findings: list[Finding]) -> dict`

#### `RootCauseReport(Report)`
Final output of the Root Cause Analysis.
- **Attributes (extends Report):** `ranked_causes: list[Hypothesis]`, `attribution_breakdown: dict`, `event_timeline: list[dict]`

---

### 11.3 Performance Ceiling Subsystem `[PRO]`

#### `PerformanceCeilingSubsystem`
Estimates the maximum achievable performance and identifies binding bottlenecks.
- **Tier:** `CapabilityTier.PRO`
- `run(ctx: InvestigationContext) -> CeilingReport`

#### `IrreducibleErrorEstimator`
Estimates the Bayes error rate using ensemble and nearest-neighbor methods.
- `estimate(df, label_col: str) -> float`

#### `BottleneckRanker`
Decomposes the gap between current performance and the ceiling.
- `rank(current_metrics: dict, ceiling: float, df, model: ModelVersion) -> list[dict]`

#### `DataVolumeSaturationProjector`
Projects whether more data would significantly improve performance.
- `project(model_id: str, version: str, historical_performance: list[tuple]) -> dict`
- `is_data_limited(current_size: int, projection: dict) -> bool`

#### `ModelAdequacyTester`
Assesses whether a larger/different model architecture could extract more signal.
- `test(model_id: str, dataset: DatasetBinding, available_versions: list[ModelVersion]) -> dict`

#### `CeilingReport(Report)`
Final output of the Performance Ceiling analysis.
- **Attributes (extends Report):** `performance_ceiling: float`, `current_gap: float`, `ranked_bottlenecks: list[dict]`, `data_limited: bool`

---

### 11.4 Incident Response & Action Recommendation Subsystem `[PRO]`

#### `IncidentResponseSubsystem`
Synthesizes all findings into a prioritized action plan.
- **Tier:** `CapabilityTier.PRO`
- `run(all_findings: list[Finding], ctx: InvestigationContext) -> IncidentResponsePlan`

#### `ActionGenerator`
Generates specific, concrete actions for each identified root cause.
- `generate(finding: Finding) -> list[Action]`

#### `Action`
A single recommended action with impact and effort estimates.
- **Attributes:** `description: str`, `type: ActionType`, `expected_impact: str`, `effort: EffortLevel`, `priority: int`, `risk_level: str`, `prerequisites: list[str]`

#### `ActionType` (enum)
- Values: `RETRAIN`, `ROLLBACK`, `RECALIBRATE`, `FIX_PIPELINE`, `ADD_DATA`, `REMOVE_FEATURE`, `THRESHOLD_ADJUST`, `ALERT_ONLY`

#### `ActionPrioritizer`
Ranks actions by expected impact, effort, and confidence.
- `prioritize(actions: list[Action], business_constraints: dict | None) -> list[Action]`

#### `WhatIfSimulator`
Projects performance improvement if a given action is taken.
- `simulate(action: Action, simulator: CounterfactualSimulator) -> dict`

#### `DependencyMapper`
Builds an execution roadmap respecting action ordering dependencies.
- `map(actions: list[Action]) -> list[list[Action]]`

#### `AutomatedActionExecutor`
Executes low-risk actions (recalibration, threshold adjustment) automatically.
- `can_automate(action: Action) -> bool`
- `execute(action: Action, approval_token: str | None) -> dict`

#### `FeedbackTracker`
Records outcomes of past recommendations to improve future ones.
- `record(action: Action, outcome: dict) -> None`
- `get_historical_effectiveness(action_type: ActionType) -> float`

#### `IncidentResponsePlan`
A complete, prioritized plan for responding to an incident.
- **Attributes:** `incident_id: UUID`, `actions: list[Action]`, `execution_roadmap: list[list[Action]]`, `projected_recovery_metrics: dict`, `auto_executable: list[Action]`

---

### 11.5 System Health Report Aggregation Subsystem

#### `HealthReportAggregationSubsystem`
Compiles all findings into a holistic `SystemHealthReport`.
- `run(all_findings: list[Finding], ctx: InvestigationContext) -> SystemHealthReport`

#### `SystemHealthReport(Report)`
Final output of the System Investigation Engine.
- **Attributes (extends Report):** `open_incidents: list[Incident]`, `drift_summary: dict`, `performance_trend: list[tuple]`

#### `SystemHealthScoreCalculator`
Computes the 0–100 system health score.
- `calculate(findings: list[Finding], open_incidents: list[Incident]) -> HealthScore`

#### `CrossSubsystemSynthesizer`
Aggregates findings across all system investigation subsystems.
- `synthesize(findings: list[Finding]) -> list[Finding]`

#### `IncidentLifecycleManager`
Tracks incidents from detection through resolution and post-mortem.
- `open_from_findings(findings: list[Finding]) -> Incident`
- `update_status(incident_id: str, status: IncidentStatus, notes: str) -> None`
- `close(incident_id: str, resolution: str) -> None`

#### `StakeholderReportGenerator`
Produces periodic (daily/weekly) narrative health summaries.
- `generate_daily(model_id: str) -> str`
- `generate_weekly(model_id: str) -> str`
- `export_pdf(model_id: str, period: str) -> bytes`

#### `ContinuousImprovementTracker`
Measures actual improvement from past recommendations.
- `record_pre_action_metrics(action_id: str, metrics: dict) -> None`
- `record_post_action_metrics(action_id: str, metrics: dict) -> None`
- `compute_effectiveness(action_id: str) -> float`

---

## 12. `holmes.adapters.*`

All adapter implementations. Each implements one or more port contracts.

### `holmes.adapters.mlflow`
| Class | Implements | Notes |
|---|---|---|
| `MLflowModelStoreAdapter` | `IModelStore` | Wraps MLflow tracking and model registry APIs |
| `MLflowRunAdapter` | — | Converts MLflow `Run` objects to Holmes `TrainingRun` |

### `holmes.adapters.wandb`
| Class | Implements | Notes |
|---|---|---|
| `WandbModelStoreAdapter` | `IModelStore` | Wraps W&B artifact and run APIs |
| `WandbArtifactAdapter` | — | Converts W&B artifacts to `DataArtifactPointer` |

### `holmes.adapters.dbt`
| Class | Implements | Notes |
|---|---|---|
| `DbtDataCatalogAdapter` | `IDataCatalog` | Reads dbt manifest.json and catalog.json |
| `DbtLineageAdapter` | — | Builds `ModelLineage` from dbt DAG |

### `holmes.adapters.kafka`
| Class | Implements | Notes |
|---|---|---|
| `KafkaEventBusAdapter` | `IEventBus` | Async event bus via Kafka. PRO tier |
| `KafkaTopicManager` | — | Topic creation and config management |

### `holmes.adapters.inprocess`
| Class | Implements | Notes |
|---|---|---|
| `InProcessEventBus` | `IEventBus` | Sync in-process pub/sub. Default for OSS |
| `AsyncInProcessEventBus` | `IEventBus` | Async variant using asyncio. OSS |

### `holmes.adapters.storage`
| Class | Implements | Notes |
|---|---|---|
| `LocalStorageAdapter` | `IStorageBackend` | Writes to local filesystem. Default for OSS |
| `S3StorageAdapter` | `IStorageBackend` | AWS S3. PRO tier |
| `GCSStorageAdapter` | `IStorageBackend` | Google Cloud Storage. PRO tier |
| `AzureBlobStorageAdapter` | `IStorageBackend` | Azure Blob Storage. PRO tier |

### `holmes.adapters.shap`
| Class | Implements | Notes |
|---|---|---|
| `ShapExplainabilityAdapter` | `IExplainabilityEngine` | Default explainer using SHAP library |

### `holmes.adapters.notification`
| Class | Implements | Notes |
|---|---|---|
| `SlackNotificationAdapter` | `INotifier` | Sends alerts to Slack channels |
| `PagerDutyNotificationAdapter` | `INotifier` | Creates PagerDuty incidents |
| `EmailNotificationAdapter` | `INotifier` | SMTP-based email reports |
| `WebhookNotificationAdapter` | `INotifier` | Generic HTTP webhook |

### `holmes.adapters.inference`
| Class | Implements | Notes |
|---|---|---|
| `LocalModelInferenceAdapter` | `IInferenceEngine` | Loads pickled sklearn/XGBoost models locally |
| `BentoMLInferenceAdapter` | `IInferenceEngine` | Calls BentoML serving API |
| `TritonInferenceAdapter` | `IInferenceEngine` | NVIDIA Triton Inference Server |
| `SageMakerInferenceAdapter` | `IInferenceEngine` | AWS SageMaker endpoint. PRO |

---

## 13. `holmes.api`

REST / GraphQL interface layer. Thin layer — no business logic here.

### Classes

| Class | Responsibility |
|---|---|
| `ApiGateway` | Main FastAPI application. Mounts all routers. Initialises DI container |
| `InvestigationRouter` | `POST /investigations/data`, `POST /investigations/model` |
| `MonitoringRouter` | `POST /monitoring/start`, `DELETE /monitoring/{model_id}` |
| `ModelRouter` | `POST /models`, `GET /models/{id}/versions` |
| `DataRouter` | `POST /datasets`, `GET /datasets/{id}/profile` |
| `FindingRouter` | `GET /findings` with query params |
| `ReportRouter` | `GET /reports/{id}`, `GET /models/{id}/reports/latest` |
| `IncidentRouter` | `GET /incidents`, `POST /incidents/{id}/resolve` |
| `AuthMiddleware` | JWT validation and tier resolution |
| `RateLimitMiddleware` | Per-tenant rate limiting |
| `RequestContext` | Per-request context (tenant, user, correlation ID) |
| `ErrorHandler` | Maps domain exceptions to HTTP responses |

---

## 14. `holmes.sdk`

Python SDK for ML developers to integrate Holmes into their training pipelines.

### Classes

| Class | Responsibility |
|---|---|
| `HolmesClient` | Main SDK entry point. Holds credentials and base URL |
| `ModelTracker` | Context manager — wraps training runs and auto-registers models |
| `PredictionLogger` | Logs production predictions to Holmes in batch or streaming |
| `DatasetRegistrar` | Registers and profiles datasets from pandas DataFrames or file paths |
| `InvestigationClient` | Triggers data/model investigations and polls for results |
| `FindingsViewer` | Retrieves and pretty-prints findings and reports |
| `HolmesCallback` | Framework callback (e.g. Keras, PyTorch Lightning) that auto-logs metrics |
