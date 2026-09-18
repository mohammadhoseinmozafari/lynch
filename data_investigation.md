We'll decompose the **Data Investigation Layer** into a set of cooperating subsystems, all part of the **Data Investigation Engine (DI Engine)**. Each subsystem is responsible for a specific category of analysis, produces structured findings, and relies on the existing Holmz services for data access, computation, and storage.

The subsystems are:

1. **Data Profiling & Integrity Subsystem**  
2. **Leakage Detection Subsystem**  
3. **Label Quality Subsystem**  
4. **Coverage Analysis Subsystem**  
5. **Signal & Predictability Subsystem**  
6. **Report Aggregation & Recommendation Subsystem**

---

## 1. Data Profiling & Integrity Subsystem

**Purpose**  
To thoroughly examine the dataset for quality issues, anomalies, and structural inconsistencies, and to produce contextualized findings that go beyond simple counts.

**Key Responsibilities**
- **Missing value analysis**  
  - Compute missing rates per feature and per row.  
  - Test whether missingness is completely at random (MCAR), at random (MAR), or not at random (MNAR) by correlating missing indicators with other features and the target.  
  - Identify clusters/segments where missingness is concentrated (e.g., “78% of missing values in `income` come from customers with `account_age < 30 days`”).  
  - Recommend imputation strategies or feature exclusion based on the missingness mechanism.

- **Duplicate detection & impact assessment**  
  - Detect exact and near‑duplicate records using fuzzy hashing and row similarity.  
  - Analyze temporal patterns (e.g., duplicates spike after a specific pipeline run).  
  - Check whether duplicates have conflicting labels; flag harmful duplicates that distort training.

- **Outlier discovery**  
  - Use multivariate distance (e.g., isolation forest, Mahalanobis) to find unusual points.  
  - Classify outliers as “valid extreme”, “potential error”, or “distributional anomaly” by consulting business rules and feature‑level profiles.  
  - For “potential errors”, suggest correction strategies (e.g., cap at 99th percentile, cross‑reference with other sources).

- **Schema & type validation**  
  - Compare actual data types and value ranges against the registered `DataSchema`.  
  - Detect new or missing columns, type changes, or value‑range violations.  
  - Track schema drift across data versions.

- **Data freshness & staleness**  
  - Compute record timestamps and identify late‑arriving data, gaps, or outdated records.  
  - Flag segments where data is too old to be reliable for training.

**Inputs**  
- `DatasetBinding` from **MEMS** (schema, profile, split pointers).  
- Actual data (via `DataArtifactPointer`) read through a data access layer.

**Outputs**  
- A set of `Finding` objects with types like `MISSING_NOT_AT_RANDOM`, `DUPLICATE_SPIKE`, `OUTLIER_CLUSTER`, `SCHEMA_DRIFT`, `STALE_DATA`.  
- Each finding contains severity, confidence, supporting evidence (samples, statistics), and a concrete recommendation.

**Integration**  
Runs automatically when a new dataset is registered or on demand. Publishes findings to **EDAS** and updates the dataset’s health status in **MEMS**.

---

## 2. Leakage Detection Subsystem

**Purpose**  
To identify features or preprocessing steps that introduce illegitimate predictive power, preventing models from learning true patterns.

**Key Responsibilities**
- **Temporal leakage testing**  
  - For each feature, determine if its values depend on information that would not be available at prediction time.  
  - For time‑series data, verify chronological integrity: simulate a “point‑in‑time” check to see if the feature’s value at time *t* uses information from after *t*.  
  - Flag features like “days since last purchase” computed after the target event.

- **Target‑informed preprocessing detection**  
  - Check if scaling, imputation, or encoding parameters were computed using the entire dataset (including validation/test) rather than only the training split.  
  - Use a “leakage audit” by re‑fitting preprocessing on a subset and comparing feature values; discrepancies indicate leakage.

- **Proxy variable identification**  
  - Discover features that are not directly causal but act as near‑perfect predictors due to data collection artifacts (e.g., `hospital_id` predicting readmission because one hospital has a policy that increases readmissions).  
  - Use domain‑agnostic metrics like “predictive power stability” – if a feature’s importance collapses when you remove a few instances or change the data source, it is likely a proxy.

- **Duplicate‑induced leakage**  
  - Detect when train/test splits contain records from the same entity (e.g., same user) leading to over‑optimistic evaluation.  
  - Compute the overlap index and recommend split strategies (e.g., group‑based splitting).

- **Leakage confidence scoring**  
  - Quantify the extent to which a feature artificially inflates validation metrics.  
  - Simulate the model’s performance without the suspect feature to estimate the “true” performance baseline.

**Inputs**  
- `DatasetBinding` (schema, split definitions, temporal metadata).  
- Optionally, a lightweight model trained via **ISS** to test feature predictiveness.  
- Statistical profiles from the Data Profiling subsystem.

**Outputs**  
- `Finding` objects with types like `TEMPORAL_LEAKAGE`, `PREPROCESSING_LEAKAGE`, `PROXY_VARIABLE`, `TRAIN_TEST_OVERLAP`.  
- Recommendations: features to remove, preprocessing to re‑run, split redesign.

**Integration**  
Runs after the Data Profiling subsystem. Can also be triggered by the **System Investigation Engine** when production performance deviates from validation.

---

## 3. Label Quality Subsystem

**Purpose**  
To assess the trustworthiness of labels and uncover systematic labeling errors that would limit model performance.

**Key Responsibilities**
- **Confidence‑based mislabel detection**  
  - Train a fast surrogate model to get predictions on training data.  
  - Apply confident‑learning techniques: flag instances where the model is very confident but disagrees with the given label.  
  - Separate model blindness from genuine label noise by checking consistency across multiple models or folds.

- **Annotator disagreement analysis**  
  - If multiple label sources exist, compute inter‑annotator agreement and find samples with high disagreement.  
  - Correlate disagreement with feature values to find segments where the labeling task is inherently ambiguous or guidelines are inconsistently applied.

- **Label drift detection**  
  - Compare label distributions across time or data batches.  
  - Detect when the definition of a class changes (e.g., “churn” changed from 60 days to 30 days).  
  - Alert if the label drift is significant enough to cause performance regression.

- **Systematic noise pattern discovery**  
  - Find whether mislabels are concentrated in certain subpopulations (e.g., elderly users, complex transactions).  
  - Output the “noise matrix” (probability of true label given observed label) per segment.

- **Prioritized re‑label list**  
  - Rank mislabeled instances by the expected impact on model performance if corrected.  
  - Estimate the cost‑benefit of re‑labeling versus using noise‑robust algorithms.

**Inputs**  
- `DatasetBinding` (labels, schema).  
- Model predictions from **ISS** (or a dedicated fast evaluator).  
- Optionally, annotator metadata if available.

**Outputs**  
- `Finding` types: `MISLABELED_INSTANCE`, `ANNOTATOR_DISAGREEMENT`, `LABEL_DRIFT`, `SYSTEMATIC_NOISE`.  
- A `LabelHealthReport` with noise estimates and a prioritized list for re‑labeling.

**Integration**  
Triggers after model registration when a model becomes available. For purely data‑centric checks, can run with a simple baseline model. Results stored in **EDAS** and linked to the dataset in **MEMS**.

---

## 4. Coverage Analysis Subsystem

**Purpose**  
To map where the training data is dense and where it is sparse, and to compare against a production or target population to expose representation gaps.

**Key Responsibilities**
- **Multidimensional density estimation**  
  - Build a density model of the training data using kernel density estimation, Gaussian mixtures, or neural density estimators.  
  - Identify low‑density regions (holes) and high‑density clusters.

- **Gap detection against reference distribution**  
  - Compare training density against a reference (production data, expected population).  
  - Detect not only marginal gaps but intersectional gaps (e.g., “young + high income + rural”).  
  - Quantify the size of each gap and the expected model error in that region.

- **Cold‑start region mapping**  
  - Define the convex hull or minimal bounding region of the training data; anything outside is an extrapolation zone.  
  - Assign an extrapolation risk score to any new instance.

- **Business‑criticality weighting**  
  - If business metrics (e.g., revenue, risk) are available, weight coverage gaps by their business impact to focus on high‑cost gaps.

- **Coverage sufficiency scoring**  
  - For each feature and feature combination, estimate whether there is enough data to learn reliable predictions (based on sample size and signal strength).

**Inputs**  
- `DataProfile` and `DataSchema` from **MEMS**.  
- Reference data (production sample) provided via **ISS** or uploaded separately.

**Outputs**  
- `Finding` types: `POPULATION_GAP`, `INTERSECTIONAL_GAP`, `EXTRAPOLATION_ZONE`, `COLD_START_REGION`.  
- A `CoverageMap` that can be visualized, showing safe and high‑risk zones.

**Integration**  
Runs after Data Profiling. Reference distribution can be taken from a production split or uploaded separately. Findings are used by the **Model Investigation Engine** to probe failure regions.

---

## 5. Signal & Predictability Subsystem

**Purpose**  
To quantify how much predictive information exists in the data, which features carry it, and what is the best performance the data can support.

**Key Responsibilities**
- **Mutual information estimation**  
  - Compute pairwise and conditional mutual information between each feature and the target.  
  - Use k‑NN or neural estimators to handle high‑dimensional, mixed‑type data.

- **Redundancy & synergy analysis**  
  - Find groups of features that encode redundant information (low unique information).  
  - Discover synergistic pairs/groups that together carry more information than individually.

- **Bayes error / irreducible error estimation**  
  - Estimate the lowest possible error rate using ensemble agreement, neighborhood methods, or optimal transport.  
  - Provide an upper bound on achievable AUC, F1, etc.

- **Signal‑to‑noise ratio by segment**  
  - Partition the data (e.g., by population segments) and estimate the predictability within each partition.  
  - Reveal segments where the target is inherently random.

- **Data‑volume saturation projection**  
  - Fit a learning curve (e.g., power‑law) to project how performance would change with more data.  
  - Identify whether the model is data‑limited or plateaued.

- **Feature value assessment**  
  - Estimate the potential performance gain from adding an external feature or improving the measurement accuracy of an existing one.

**Inputs**  
- `DatasetBinding` (features, target, splits).  
- Optionally, a trained model to measure performance on subsets.

**Outputs**  
- `Finding` types: `LOW_SIGNAL`, `FEATURE_REDUNDANCY`, `BAYES_ERROR_ESTIMATE`, `DATA_LIMITED`, `SIGNAL_HETEROGENEITY`.  
- A `SignalReport` with learning curves and improvement recommendations.

**Integration**  
Runs after Data Profiling. Can be used to inform whether further data collection or feature engineering is worthwhile. Outputs feed into the **Model Investigation Engine** for performance ceiling analysis.

---

## 6. Report Aggregation & Recommendation Subsystem

**Purpose**  
To synthesize raw findings from the other subsystems into a coherent, prioritized `DataHealthReport`, and to generate actionable recommendations.

**Key Responsibilities**
- **Finding deduplication & grouping**  
  - Merge related findings (e.g., missing values and outliers that co‑occur in a segment).  
  - Create a unified story: “The ‘new customers’ segment has high missingness in `income`, is underrepresented, and has noisy labels.”

- **Severity & confidence normalization**  
  - Standardize severity and confidence scores across subsystems so they are comparable.  
  - Compute an overall data health score (0–100) as a quick indicator.

- **Action prioritization**  
  - Rank recommendations by expected impact (performance gain, risk reduction), effort, and confidence.  
  - Use a cost‑benefit heuristic or a simple scoring model.

- **Report generation**  
  - Compile the `DataHealthReport` with executive summary, detailed findings, and recommendations.  
  - Store the report in **EDAS** and link it to the dataset version in **MEMS**.

- **Feedback loop**  
  - Allow users to accept or dismiss findings; use this feedback to improve future investigations (e.g., if a user repeatedly dismisses a certain type of finding as irrelevant, the system can deprioritize it).

**Inputs**  
- All findings from the other subsystems.  
- Business constraints (e.g., acceptable risk thresholds) from project configuration.

**Outputs**  
- `DataHealthReport` (aggregated).  
- Updated dataset health status in **MEMS**.

**Integration**  
Runs after all other subsystems have completed. Provides the final output that is presented to the user via the DIL.

---

## Subsystem Interaction Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   DATA INVESTIGATION ENGINE                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 1. Profiling │  │ 2. Leakage   │  │ 3. Label     │          │
│  │ & Integrity  │  │ Detection    │  │ Quality      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                        │
│  ┌──────────────┐            ┌──────────────┐                  │
│  │ 4. Coverage  │            │ 5. Signal &  │                  │
│  │ Analysis     │            │ Predictability│                  │
│  └──────┬───────┘            └──────┬───────┘                  │
│         │                           │                           │
│         └─────────────┬─────────────┘                           │
│                       ▼                                         │
│            ┌─────────────────────────┐                          │
│            │ 6. Report Aggregation   │                          │
│            │ & Recommendation        │                          │
│            └─────────────────────────┘                          │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
          ┌─────────────────────────────┐
          │        EDAS                  │
          │ (Findings & Health Reports)  │
          └─────────────────────────────┘
```

All subsystems read from **MEMS** (dataset metadata) and can call **ISS** for lightweight model inference when needed (e.g., Label Quality uses a model to find mislabels, Leakage Detection may test feature predictiveness). They publish findings to **EDAS**. The entire engine can be triggered by a dataset registration event or manually. This modular design allows each subsystem to be developed, tested, and scaled independently, while the shared finding schema ensures consistency.