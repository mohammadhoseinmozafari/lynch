We'll decompose the **System Investigation Layer** into a set of interconnected subsystems that together form the **System Investigation Engine**. This layer monitors production behavior, detects performance degradation, isolates root causes, estimates performance ceilings, and generates prioritized action plans—all while leveraging the existing Holmz services.

The subsystems are:

1. **Production Drift Monitoring & Detection Subsystem**  
2. **Root Cause Analysis Subsystem**  
3. **Performance Ceiling & Bottleneck Analysis Subsystem**  
4. **Incident Response & Action Recommendation Subsystem**  
5. **System Health Report Aggregation Subsystem**

---

## 1. Production Drift Monitoring & Detection Subsystem

**Purpose**  
To continuously monitor production predictions and data streams, detect meaningful shifts in model behavior, data distributions, or performance metrics, and trigger investigations when drift exceeds acceptable thresholds.

**Key Responsibilities**
- **Data drift detection**  
  - Compare production feature distributions against training baselines (from `DataProfile` in **MEMS**) using statistical tests (Kolmogorov–Smirnov, Jensen–Shannon distance, etc.).  
  - Detect both marginal drift (per‑feature) and joint distribution drift (multivariate).  
  - Apply temporal decomposition to distinguish gradual trends from sudden jumps.

- **Prediction drift detection**  
  - Track the distribution of predictions (scores, classes) over time; compare to historical baselines.  
  - Detect shifts in prediction confidence, class balance, or uncertainty.

- **Performance monitoring**  
  - When ground truth labels become available (delayed), compute key metrics (accuracy, AUC, F1, etc.) and compare against validation baselines and previous production windows.  
  - Detect performance degradation in real‑time slices (e.g., by hour, by region).

- **Anomaly‑based drift triggers**  
  - Use unsupervised anomaly detection on the model’s internal representations or output distributions to identify unusual production periods without waiting for labels.

- **Threshold management & alerting**  
  - Allow users to set custom drift thresholds (global and per‑segment).  
  - Generate alerts (findings) when drift exceeds thresholds, with severity based on magnitude and business impact.

**Inputs**  
- Production data streams (via **ISS** event bus).  
- Training data profiles and baselines from **MEMS**.  
- Delayed ground truth labels (optional).  
- Explanation drift reports from **EOS** (if periodic global explanations are generated).

**Outputs**  
- `Finding` types: `DATA_DRIFT`, `PREDICTION_DRIFT`, `PERFORMANCE_DEGRADATION`, `CONFIDENCE_SHIFT`.  
- Drift dashboards and trend data stored in **EDAS** for querying.  
- Trigger signals for the Root Cause Analysis Subsystem.

**Integration**  
Runs continuously as a streaming service, consuming `PredictionCompleted` events from **ISS**. Publishes drift findings to **EDAS**. When a significant drift is detected, it automatically triggers the **Root Cause Analysis Subsystem**.

---

## 2. Root Cause Analysis Subsystem

**Purpose**  
When a performance issue or drift is detected, this subsystem systematically investigates potential causes, ranks them by likelihood, and provides supporting evidence. It is the core diagnostic engine of the System Investigation layer.

**Key Responsibilities**
- **Hypothesis generation**  
  - Automatically generate a list of plausible causes: data drift (per‑feature), label drift, upstream pipeline failures, model staleness, population shifts, concept drift.  
  - Use causal graphs (if available) to trace back from the observed effect.

- **Attribution of performance change**  
  - Decompose the total performance drop into contributions from individual features, subpopulations, or data segments.  
  - For example, "62% of the AUC drop is due to a shift in `merchant_category`, 18% to `transaction_amount`, 20% to a new unmodeled category."

- **Causal validation via counterfactuals**  
  - Use **ISS** simulation capabilities to test "what if" scenarios:  
    - What if the production data still followed the training distribution?  
    - What if a specific feature had not shifted?  
  - Compare predicted outcomes under counterfactual scenarios to quantify impact.

- **Timeline reconstruction**  
  - Correlate the onset of performance degradation with external events (deployments, data pipeline changes, upstream service updates, marketing campaigns).  
  - Provide an event‑attribution timeline that pinpoints when each contributing factor started.

- **Blame decomposition across the ML pipeline**  
  - Separate the degradation into components caused by data changes vs. model changes vs. environment changes.  
  - If multiple models are chained, identify which model introduced the error.

- **Confidence scoring of root causes**  
  - Assign a confidence score to each identified root cause, based on the consistency of evidence, statistical significance, and counterfactual validation.

**Inputs**  
- Drift alerts and performance metrics from the Drift Monitoring Subsystem.  
- Production data samples and predictions from **ISS**.  
- Training baselines and model lineage from **MEMS**.  
- Explanations from **EOS** for comparing feature attributions before/after drift.  
- Deployment history and pipeline metadata (can be stored in MEMS or external systems).

**Outputs**  
- `Finding` types: `ROOT_CAUSE_IDENTIFIED`, `PERFORMANCE_ATTRIBUTION`, `EVENT_TIMELINE`, `CAUSAL_VALIDATION`.  
- A `RootCauseReport` with ranked causes, supporting evidence, and confidence levels.

**Integration**  
Triggered by the Drift Monitoring Subsystem or manually by a user. Calls **ISS** for counterfactual simulations and **EOS** for attribution comparisons. Stores findings and reports in **EDAS**, linked to the model version and time window.

---

## 3. Performance Ceiling & Bottleneck Analysis Subsystem

**Purpose**  
To estimate the maximum achievable performance under current constraints and identify the binding bottlenecks that limit further improvement.

**Key Responsibilities**
- **Irreducible error estimation**  
  - Estimate the Bayes error rate (or equivalent) using ensemble methods, density estimation, or nearest‑neighbor techniques on the production data distribution.  
  - Provide a realistic upper performance bound given the data quality and feature set.

- **Bottleneck ranking**  
  - Decompose the gap between current performance and the ceiling into components:  
    - Label noise / inherent randomness  
    - Missing features or insufficient feature informativeness  
    - Model capacity limitations  
    - Data volume insufficiency  
    - Distribution shift (the model was not trained on the current production distribution)  
  - Rank these components by their contribution to the gap.

- **Data‑volume saturation projection**  
  - Fit learning curves on historical production data (if available) to project whether collecting more production data would significantly improve performance.  
  - Identify if the model is in a data‑limited regime.

- **Feature value assessment**  
  - Analyze whether improving the quality or availability of specific features (e.g., reducing missingness, adding new sensors) would lift the performance ceiling.  
  - Estimate the potential gain per feature.

- **Model adequacy test**  
  - Determine if a larger or different model architecture could extract more signal from the existing data.  
  - Use scaling laws or architectural comparisons (if multiple model versions are registered in MEMS) to assess this.

**Inputs**  
- Production data and labels (if available) from **ISS** or direct feeds.  
- Model metadata and training data profiles from **MEMS**.  
- Baseline performance metrics from validation (stored in MEMS).  
- Signal analysis results from the Data Investigation Engine (if available).

**Outputs**  
- `Finding` types: `PERFORMANCE_CEILING`, `DATA_LIMITED`, `FEATURE_BOTTLENECK`, `MODEL_CAPACITY_LIMIT`.  
- A `CeilingReport` showing the performance headroom, ranked bottlenecks, and recommended directions for improvement.

**Integration**  
Runs periodically or on demand, often after a root cause analysis finds that the model is not broken but underperforming due to fundamental limitations. Uses **ISS** for simulations and **MEMS** for lineage. Stores reports in **EDAS**.

---

## 4. Incident Response & Action Recommendation Subsystem

**Purpose**  
To synthesize diagnostic findings into a concrete, prioritized action plan for resolving production issues or improving system performance. It acts as the decision‑support layer that guides the team from insight to action.

**Key Responsibilities**
- **Action generation**  
  - For each root cause or bottleneck, generate one or more specific actions:  
    - "Retrain model with up‑to‑date data"  
    - "Roll back model to version X"  
    - "Fix data pipeline bug in feature Y"  
    - "Add more training data for segment Z"  
    - "Apply recalibration to restore confidence reliability"

- **Prioritization**  
  - Rank actions by expected impact (performance gain, risk reduction), implementation effort (low/medium/high), and confidence in the diagnosis.  
  - Use a cost‑benefit scoring model that can incorporate business constraints.

- **What‑if simulation**  
  - Allow users to ask "What if I take action A?" and see the projected performance improvement and risk reduction, based on counterfactual models.  
  - Example: "If you retrain on the last 30 days of data, expected AUC improvement is +0.04."

- **Dependency mapping**  
  - Identify actions that must precede others (e.g., fix the data pipeline before retraining).  
  - Build an execution roadmap.

- **Automated triggers**  
  - For low‑risk actions (e.g., recalibration, threshold adjustment), offer one‑click or even fully automated execution.  
  - For high‑risk actions (e.g., model rollback), generate a detailed rollback plan with approval gates.

- **Feedback loop**  
  - Track which actions were taken and their actual impact; feed this back to improve future recommendations (learning from past incidents).

**Inputs**  
- All findings from the Drift Monitoring, Root Cause Analysis, and Performance Ceiling subsystems.  
- Model deployment history and pipeline metadata from **MEMS**.  
- Business impact metrics (e.g., revenue loss due to model errors) optionally provided.

**Outputs**  
- `Finding` types: `RECOMMENDED_ACTION`, `PROJECTED_IMPACT`, `AUTOMATED_FIX`.  
- An `IncidentResponsePlan` document with prioritized actions, timelines, and expected outcomes.

**Integration**  
Runs after Root Cause Analysis or Performance Ceiling analysis. Uses **ISS** for what‑if simulations. Can interact with external deployment tools (via webhooks) for automated rollback or retraining. Stores recommendations in **EDAS** and displays them in DIL.

---

## 5. System Health Report Aggregation Subsystem

**Purpose**  
To compile all findings from the System Investigation layer into a holistic view of system health, track historical incidents, and provide a continuously updated status for stakeholders.

**Key Responsibilities**
- **Cross‑subsystem synthesis**  
  - Aggregate findings from Drift Monitoring, Root Cause Analysis, and Performance Ceiling subsystems into a unified timeline and summary.  
  - Identify correlations between different incidents (e.g., recurring drift patterns).

- **System health scoring**  
  - Compute an overall System Health Score (0–100) based on:  
    - Current performance relative to baseline  
    - Drift severity  
    - Open incidents  
    - Performance headroom  
  - Track this score over time to show trends.

- **Incident lifecycle management**  
  - Track each incident from detection → diagnosis → action → resolution → post‑mortem.  
  - Maintain an incident database in **EDAS** for historical analysis.

- **Stakeholder reporting**  
  - Generate periodic (daily/weekly) health reports with key metrics, active incidents, and recommended actions.  
  - Support export and integration with external monitoring dashboards.

- **Continuous improvement tracking**  
  - Measure the effectiveness of past recommendations by comparing pre‑ and post‑action performance.  
  - Use this data to refine the recommendation engine’s models.

**Inputs**  
- All findings and reports from other System Investigation subsystems.  
- Historical performance baselines and deployment records from **MEMS**.  
- Feedback from users on recommended actions (accepted/rejected/impact).

**Outputs**  
- `SystemHealthReport` (periodic summary).  
- System health score and trend data.  
- Incident history and resolution metrics.

**Integration**  
Runs as a periodic aggregator and on‑demand report generator. Pulls data from **EDAS** (findings store) and **MEMS** (model performance history). Serves as the primary data source for DIL’s monitoring dashboard.

---

## Subsystem Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                   SYSTEM INVESTIGATION ENGINE                      │
│                                                                    │
│  ┌──────────────────────────┐                                     │
│  │ 1. Drift Monitoring &    │                                     │
│  │    Detection             │                                     │
│  └───────────┬──────────────┘                                     │
│              │ triggers                                             │
│              ▼                                                     │
│  ┌──────────────────────────┐      ┌──────────────────────────┐   │
│  │ 2. Root Cause Analysis   │      │ 3. Performance Ceiling & │   │
│  │                          │      │    Bottleneck Analysis   │   │
│  └───────────┬──────────────┘      └───────────┬──────────────┘   │
│              │                                 │                   │
│              └─────────────┬───────────────────┘                   │
│                            ▼                                       │
│            ┌──────────────────────────────┐                        │
│            │ 4. Incident Response & Action│                        │
│            │    Recommendation            │                        │
│            └──────────────┬───────────────┘                        │
│                           │                                        │
│                           ▼                                        │
│            ┌──────────────────────────────┐                        │
│            │ 5. System Health Report      │                        │
│            │    Aggregation               │                        │
│            └──────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
          ┌─────────────────────────────┐
          │        EDAS                  │
          │ (Findings, Reports, Incidents│
          │  & Health Scores)            │
          └─────────────────────────────┘
```

Each subsystem leverages **ISS** for inference and simulations, **MEMS** for baselines and lineage, **EOS** for explanations when comparing feature behavior, and **EDAS** as the unified storage for findings and reports. The entire engine runs as a continuous background process, with the Incident Response Subsystem providing decision support and automation capabilities. This modular architecture allows each subsystem to evolve independently while sharing a common finding schema.