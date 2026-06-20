We'll decompose the **Model Investigation Layer** into a set of specialized subsystems that form the **Model Investigation Engine**. Each subsystem examines a distinct aspect of model trustworthiness, produces structured findings, and leverages existing Holmz services for computation, lineage, and storage.

The subsystems are:

1. **Generalization & Robustness Subsystem**  
2. **Failure Region Discovery Subsystem**  
3. **Calibration & Confidence Subsystem**  
4. **Shortcut Learning Detection Subsystem**  
5. **Prediction Stability Subsystem**  
6. **Feature Dependency & Interaction Subsystem**  
7. **Model Health Report Aggregation & Recommendation Subsystem**

---

## 1. Generalization & Robustness Subsystem

**Purpose**  
To assess how well the model performs beyond its training distribution and to quantify its robustness against various forms of distribution shift.

**Key Responsibilities**
- **Multi‑faceted shift simulation**  
  - Apply covariate shift, label shift, and concept drift simulations using perturbation functions (e.g., adding Gaussian noise, applying monotonic transformations, reweighting samples).  
  - Generate synthetic out‑of‑distribution (OOD) data points by interpolating between training samples or sampling from low‑density regions (identified by the Data Investigation Engine).

- **Robustness curve construction**  
  - For each type of shift, measure performance degradation as a function of shift magnitude.  
  - Identify the maximum “safe” shift before performance drops below an acceptable threshold.

- **Adversarial vulnerability profiling**  
  - Run adversarial attacks (e.g., FGSM, PGD for differentiable models, or query‑based attacks for black‑box models) to discover small input perturbations that cause large prediction changes.  
  - Map the frequency and severity of adversarial examples across the input space.

- **OOD detection boundary definition**  
  - Train or calibrate an OOD detector on the model’s internal representations (e.g., Mahalanobis distance in feature space) or prediction uncertainties.  
  - Define a decision boundary in the OOD score that separates reliable predictions from unreliable ones; this boundary becomes a deployment guardrail.

- **Cross‑validation robustness**  
  - When multiple model versions exist (e.g., from different training runs), compare their robustness profiles to identify the most resilient candidate.

**Inputs**  
- Model artifact and metadata from **MEMS**.  
- Training data distribution (from **DataProfile** in MEMS) and, optionally, production data samples.  
- Inference via **ISS** to obtain predictions and optionally gradients/activations.  
- Explanation generation from **EOS** for attribution under perturbations.

**Outputs**  
- `Finding` types: `ROBUSTNESS_PROFILE`, `ADVERSARIAL_VULNERABILITY`, `OOD_BOUNDARY`, `DISTRIBUTION_SHIFT_TOLERANCE`.  
- A `RobustnessReport` containing robustness curves, safe‑shift ranges, and recommended guardrails.

**Integration**  
Runs after model registration and as part of pre‑deployment checks. Can be triggered periodically when production data shifts. Stores results in **EDAS** and updates model health in **MEMS**.

---

## 2. Failure Region Discovery Subsystem

**Purpose**  
To automatically identify subpopulations or slices where the model performs significantly worse than average, including hidden intersections not captured by predefined categories.

**Key Responsibilities**
- **Slice‑finding algorithm**  
  - Search across all possible feature conditions (both single‑feature and multi‑feature combinations) to find slices with high error concentration.  
  - Use techniques like decision‑tree‑based slice discovery, beam search, or statistical testing to surface the most critical failures.

- **Slice characterization**  
  - For each discovered failure region, compute: error rate, size (number of instances), trend over time (if temporal data available), and business impact (if labels or costs attached).  
  - Determine whether the failures are systematic (consistent across models) or model‑specific.

- **Slice‑level feature attribution**  
  - Generate explanations (via **EOS**) specifically for the instances within a failure slice.  
  - Compare feature importance inside the slice versus globally to understand what drives errors there.

- **Emergent failure monitoring**  
  - Track slice performance in production; detect when a previously benign slice becomes problematic (drift in error rate).  
  - Alert when new failure regions appear.

- **Fairness‑relevant slice identification**  
  - If protected attributes are specified, automatically flag slices that indicate potential bias or disparate impact.

**Inputs**  
- Model predictions on validation/test/production data from **ISS**.  
- Ground truth labels (from dataset bindings).  
- Feature definitions and metadata from **MEMS**.  
- Explanations from **EOS**.

**Outputs**  
- `Finding` types: `FAILURE_SLICE`, `EMERGENT_FAILURE`, `BIAS_INDICATOR`.  
- A `FailureRegionReport` with ranked slices, each detailing error statistics, feature attributions, and suggested remediation.

**Integration**  
Runs after model training/registration and continuously in production. Uses **ISS** for batch inference. Findings linked to model version in **MEMS** and stored in **EDAS**.

---

## 3. Calibration & Confidence Subsystem

**Purpose**  
To evaluate whether predicted probabilities reflect true likelihoods, and to detect over‑ or under‑confidence across the prediction space and within specific subpopulations.

**Key Responsibilities**
- **Expected calibration error (ECE) computation**  
  - Calculate ECE and related metrics (MCE, Brier score) globally and within user‑defined slices.  
  - Generate reliability diagrams with confidence intervals.

- **Subpopulation calibration analysis**  
  - Identify segments (by feature value ranges or combinations) where calibration is significantly worse.  
  - Quantify whether miscalibration leads to over‑ or under‑confidence, and in which direction.

- **Confidence‑threshold optimization**  
  - Given a business utility function (e.g., cost of false positives vs. false negatives), compute the optimal probability threshold for binary decisions.  
  - Account for miscalibration: adjust thresholds per segment if calibration varies.

- **Calibration drift monitoring**  
  - In production, track calibration over time and alert when drift exceeds a critical level.  
  - Correlate calibration drift with feature drift (from **Data Investigation** or **System Investigation**) to suggest causes.

- **Confidence‑error correlation**  
  - Check whether high‑confidence errors cluster in specific feature regions, providing clues about model overconfidence.

**Inputs**  
- Model predictions (probabilities) from **ISS**.  
- Ground truth labels.  
- Feature metadata from **MEMS**.  
- Optional business cost matrix.

**Outputs**  
- `Finding` types: `CALIBRATION_ERROR`, `OVERCONFIDENCE`, `UNDERCONFIDENCE`, `CALIBRATION_DRIFT`.  
- A `CalibrationReport` with reliability diagrams, optimal thresholds, and segment‑specific calibration states.

**Integration**  
Runs after model training and regularly in production. Feeds into the System Investigation Engine’s root cause analysis. Stores reports in **EDAS**.

---

## 4. Shortcut Learning Detection Subsystem

**Purpose**  
To identify features or patterns that the model exploits as spurious correlations rather than genuine causal relationships, leading to brittle predictions.

**Key Responsibilities**
- **Spurious feature discovery**  
  - For each feature, measure its importance stability across different data environments (e.g., train vs. test, different time periods, different data sources).  
  - Flag features whose importance drops sharply when the environment changes as potential shortcuts.

- **Counterfactual feature removal**  
  - Retrain or fine‑tune a model without the suspect feature (or with it randomized) via **ISS** and measure performance change. A large drop suggests the model over‑relied on it.

- **Domain‑knowledge injection**  
  - Allow users to specify “suspect” or “protected” features; the subsystem tests if the model depends on them unexpectedly.  
  - Use causal graphs (if available) to test whether the model’s feature dependencies align with causal pathways.

- **Shortcut strength quantification**  
  - Compute a shortcut reliance score per feature (and per feature combination) based on stability, adversarial testing, and counterfactual simulations.

- **Simulated failure under shortcut removal**  
  - Estimate production performance if the shortcut were eliminated, guiding mitigation (e.g., data augmentation, regularization).

**Inputs**  
- Model and training data from **MEMS**.  
- Multiple environments or temporal splits (from dataset bindings).  
- Explanations from **EOS** (to assess feature importance stability).  
- Inference from **ISS** for counterfactual testing.

**Outputs**  
- `Finding` types: `SHORTCUT_FEATURE`, `SPURIOUS_CORRELATION`, `CAUSAL_MISMATCH`.  
- A `ShortcutReport` with ranked features by shortcut risk, recommended actions (remove, augment, regularize), and projected performance after mitigation.

**Integration**  
Runs after model registration when multiple data environments are available. Uses **EOS** for importance analysis and **ISS** for retraining simulations. Findings stored in **EDAS**.

---

## 5. Prediction Stability Subsystem

**Purpose**  
To measure the consistency of predictions under small input perturbations, random seed variations, and model retraining, distinguishing inherent uncertainty from model fragility.

**Key Responsibilities**
- **Local perturbation stability**  
  - For each instance, apply small random noise (within expected feature ranges) or dropout at inference time, and measure prediction variance.  
  - Identify instances with high fragility (prediction flips easily).

- **Decision boundary proximity**  
  - Estimate the distance of each instance to the model’s decision boundary (e.g., using adversarial perturbation methods or gradient‑based measures).  
  - Flag instances close to the boundary as inherently ambiguous.

- **Training‑variance assessment**  
  - If multiple models trained with different random seeds are available (or can be quickly fine‑tuned), compare their predictions on the same data.  
  - Disagreement among these models indicates uncertainty due to random training variation.

- **Explanation stability**  
  - Generate explanations (via **EOS**) for the same instance across multiple perturbation runs or different models; measure how much feature importance rankings change.  
  - Unstable explanations erode user trust; flag when instability exceeds a threshold.

- **Stability segmentation**  
  - Aggregate fragility scores across subpopulations to find groups where predictions are systematically unstable.

**Inputs**  
- Model(s) from **MEMS**.  
- Inference from **ISS** with perturbation support.  
- Explanations from **EOS**.

**Outputs**  
- `Finding` types: `FRAGILE_PREDICTION`, `DECISION_BOUNDARY_PROXIMITY`, `UNSTABLE_EXPLANATION`, `TRAINING_VARIANCE`.  
- A `StabilityReport` with fragility maps, uncertainty estimates, and recommendations (e.g., ensemble, more data in unstable regions).

**Integration**  
Runs after model registration and on demand. Can feed into the System Investigation when prediction flips are observed in production. Stores in **EDAS**.

---

## 6. Feature Dependency & Interaction Subsystem

**Purpose**  
To uncover complex relationships among features within the model, including interactions, suppression effects, and mediation chains, moving beyond simple importance rankings.

**Key Responsibilities**
- **Interaction strength quantification**  
  - Use Shapley interaction values, H‑statistics, or partial dependence decomposition to measure pairwise and higher‑order interactions.  
  - Build an interaction graph where nodes are features and edges represent interaction strength and direction (synergistic, redundant, antagonistic).

- **Suppression & masking detection**  
  - Identify cases where a feature’s direct importance is low, but it strongly modulates the importance of another feature (e.g., feature A suppresses feature B’s impact).  
  - Remove or alter suppressors to reveal hidden dependencies.

- **Feature clustering**  
  - Group features into functional modules based on their interaction patterns (e.g., “payment history”, “demographics”).  
  - Assess whether the model treats these modules coherently or if there are unexpected cross‑module interactions.

- **Mediation analysis**  
  - For a given feature, determine whether its influence on the prediction is direct or mediated through other features (e.g., `age` affects `income`, which then affects prediction).  
  - Requires a partial causal ordering (can be user‑provided or inferred from time precedence).

- **Context‑dependent importance**  
  - Show how a feature’s importance varies as a function of another feature’s value (e.g., “income matters more for young applicants”).  
  - Generate personalized explanation templates.

**Inputs**  
- Model and data from **MEMS**.  
- Explanations (Shapley values) from **EOS** for interaction computation.  
- Optional causal graph or temporal feature ordering.

**Outputs**  
- `Finding` types: `FEATURE_INTERACTION`, `SUPPRESSION_EFFECT`, `MEDIATION_PATH`, `CONTEXT_DEPENDENCE`.  
- An `InteractionReport` with interaction graph, suppression list, and per‑feature context profiles.

**Integration**  
Runs after model registration, using **EOS** for interaction values. Results aid in feature engineering and fairness audits. Stored in **EDAS**.

---

## 7. Model Health Report Aggregation & Recommendation Subsystem

**Purpose**  
To synthesize findings from all Model Investigation subsystems into a comprehensive `ModelHealthReport`, and to generate a prioritized action plan.

**Key Responsibilities**
- **Cross‑finding synthesis**  
  - Combine related findings: e.g., a failure region may also be a shortcut‑driven slice with unstable predictions.  
  - Build a coherent narrative of the model’s weaknesses.

- **Health scoring**  
  - Compute an overall model health score (0–100) from individual subsystem metrics (robustness, calibration, stability, etc.), weighted by business importance.

- **Recommendation engine**  
  - Generate concrete, ranked actions: “Re‑train with data augmentation for region X”, “Apply temperature scaling to fix calibration”, “Remove feature Y due to shortcut”.  
  - Each action includes estimated impact (performance lift, risk reduction), effort, and implementation guidance.

- **Deployment readiness gate**  
  - Compare the model’s health score and specific findings against deployment criteria (e.g., maximum ECE, minimum robustness threshold).  
  - Issue a go/no‑go recommendation for production.

- **Feedback incorporation**  
  - Allow users to mark findings as helpful or irrelevant; use this to tune future investigations and recommendations.

**Inputs**  
- All findings from Model Investigation subsystems (stored in **EDAS**).  
- Model metadata and performance metrics from **MEMS**.  
- Business requirements (deployment thresholds, acceptable risk).

**Outputs**  
- `ModelHealthReport` (aggregated).  
- Updated model health status in **MEMS**.  
- A prioritized action list for the team.

**Integration**  
Runs after all Model Investigation subsystems complete. Serves as the primary output to the DIL and can gate model deployment. The report is stored in **EDAS** and linked to the model version in **MEMS**.

---

## Subsystem Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                   MODEL INVESTIGATION ENGINE                       │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ 1. Generaliz-│  │ 2. Failure   │  │ 3. Calibration│            │
│  │ ation &      │  │ Region Disc. │  │ & Confidence │            │
│  │ Robustness   │  │              │  │              │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                 │                     │
│         └─────────────────┼─────────────────┘                     │
│                           │                                       │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐            │
│  │ 4. Shortcut  │  │ 5. Prediction│  │ 6. Feature   │            │
│  │ Learning     │  │ Stability    │  │ Dependency & │            │
│  │ Detection    │  │              │  │ Interaction  │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                 │                     │
│         └─────────────────┼─────────────────┘                     │
│                           │                                       │
│                           ▼                                       │
│            ┌─────────────────────────────┐                        │
│            │ 7. Report Aggregation &     │                        │
│            │    Recommendation           │                        │
│            └─────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
          ┌─────────────────────────────┐
          │        EDAS                  │
          │ (Findings & Health Reports)  │
          └─────────────────────────────┘
```

All subsystems read model metadata and data from **MEMS**, use **ISS** for inference and counterfactual simulations, leverage **EOS** for feature‑level explanations and interaction values, and publish their findings to **EDAS**. The Model Health Report Aggregation subsystem then compiles a final report that guides deployment decisions. This modular design allows incremental development and independent scaling of each investigation capability.