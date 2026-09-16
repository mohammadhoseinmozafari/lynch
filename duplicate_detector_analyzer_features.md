# Duplicate Detection — Analyzer Features Specification

## Purpose

This document defines what expert data scientists actually do
when they investigate duplicates in an ML dataset, translated
into concrete analyzer capabilities for the Holmes duplicate
detection module. Every feature listed here maps to a real,
named collector or signal extractor — not aspirational
functionality.

The governing question for every feature: **does this change
what a data scientist would do, or what they would conclude
about root cause?** If not, it doesn't belong here.

---

## What expert data scientists actually investigate

### 1. Establish what "duplicate" even means for this dataset

This is the step most tools skip entirely, and it's the most
important one. Before running any deduplication algorithm, a
senior data scientist asks: what's the key? For a customer
churn dataset, is a "duplicate" two rows with the same
customer_id? The same (customer_id, date) pair? Or two rows
that are identical across every feature? These are three
completely different claims with completely different root
causes and completely different remediation strategies.

**What this means for the analyzer:**
The module cannot assume one definition of "duplicate." It
must support multiple granularities simultaneously and report
findings separately per granularity — because a dataset can
have exact row duplicates (a copy-paste bug) while also having
entity-level duplicates (the same customer appearing in both
train and test) and neither implies the other.

---

### 2. Exact duplicate detection

The baseline check. Two rows are identical across every column.
Senior data scientists do this first because it's cheap, fully
deterministic, and its presence is almost always a pipeline bug
rather than a data characteristic — no legitimate ML dataset
should have two perfectly identical rows, and the rate alone
tells you something about severity before you investigate
further.

**What experts check here:**
- Overall exact duplicate rate (how many rows are affected)
- Whether duplicates appear across splits (train/test overlap)
  or only within one split (within-train duplicates have
  different implications than train-test overlap)
- Whether duplicates have consistent or conflicting labels —
  two identical rows with different labels are materially
  worse than two identical rows with the same label, because
  they actively corrupt gradient-based training
- Temporal clustering: do exact duplicates spike at a specific
  time, batch ID, or pipeline run? A spike points to a
  specific upstream event rather than a structural bug

**Key distinction experts make:** exact row duplicates within
train are bad but recoverable (drop one). Exact duplicates
across train and test are a validity problem — the model has
already "seen" its test data, and any evaluation metric is
meaningless.

---

### 3. Key-based / entity-level duplicate detection

A row might differ from another row in one or two columns but
refer to the same real-world entity (same customer, same
transaction, same patient). Expert data scientists identify the
natural key of the dataset (or ask the user to specify it) and
check whether that key is truly unique.

**What experts check here:**
- Key uniqueness rate: what fraction of key values appear more
  than once
- Key collision rate per split: does the same entity appear in
  both train and test (the "entity leakage" problem, which is
  structurally distinct from label leakage but similarly
  invalidates evaluation)
- Temporal overlap of the same entity across time-based splits:
  for time-series models, the same entity's history in train
  and its future in test is expected and correct; the same
  entity's *simultaneous* appearance in both is not
- Whether colliding keys have consistent feature values
  (suggesting a real duplicate) or diverging values (suggesting
  a schema change, a merge error, or a legitimate entity that
  changed over time)

**Why this matters for root cause:** key-level duplicates
across train/test are one of the most common causes of
suspiciously high validation metrics in real production ML
systems. The model achieves 0.97 AUC in validation but 0.71
in production — and the reason is that its "test" set contained
records from the same customers it trained on, which it had
memorized rather than generalized from.

---

### 4. Near-duplicate / fuzzy duplicate detection

Two rows that are "almost" identical — differ in one or two
columns, or differ only in continuous features by small amounts.
Expert data scientists check this because near-duplicates are
often more insidious than exact duplicates: they pass naive
uniqueness checks, but they still corrupt evaluation by putting
nearly identical records in both train and test, inflating
performance metrics for the same fundamental reason as exact
cross-split duplicates.

**What experts check here:**
- Pairwise similarity above a configurable threshold (using
  Jaccard similarity for categorical columns, normalized
  Euclidean distance for numerical, or Levenshtein for text)
- Whether near-duplicate pairs are within-split or cross-split
  (same priority ordering as exact duplicates: cross-split is
  the more critical finding)
- Which columns differ between near-duplicate pairs: if the
  only differing column is the label, that's a mislabeling
  signal masquerading as a near-duplicate, not a pipeline bug
- Near-duplicate rate vs. exact duplicate rate as a ratio: a
  dataset with many near-duplicates but few exact duplicates
  suggests augmentation artifacts or measurement noise rather
  than a copy-paste bug

**Computational note for the module:** pairwise comparison is
O(n²) and must be approximated at scale. Expert data scientists
use blocking (only compare rows that share a key column prefix
or hash bucket) or MinHash/LSH. The module should support both
exact (small datasets) and approximate (large datasets) with
explicit documentation of what's being traded off.

---

### 5. Label consistency analysis on duplicate pairs

This is the check that separates the diagnosis from the simple
detection. When duplicates (exact or near) have different
labels, the dataset has actively contradictory training signal:
the model is being simultaneously told "this input means Y=0"
and "this input means Y=1." This inflates the Bayes error rate
(the theoretical minimum error for this dataset) — no model
can do better than 50% on pairs it can't distinguish.

**What experts check here:**
- Label flip rate: among all duplicate pairs, what fraction
  have different labels
- Label flip rate by duplicate type (exact vs. near-duplicate):
  near-duplicates with different labels are more common and
  more ambiguous than exact duplicates with different labels
  (the latter is almost always a data bug; the former might
  be a genuine boundary case)
- Distribution of conflicting labels: is it uniformly random
  (suggesting random labeling error) or systematic (one
  specific class is always the "wrong" one, suggesting a
  labeling policy that changed mid-dataset)
- Expected performance ceiling impact: given the label flip
  rate on duplicate pairs, estimate the minimum achievable
  error this introduces

**Why this is a separate analysis and not just a property of
the duplicate rows:** expert data scientists report label
consistency findings independently from duplicate presence
findings, because the remediation differs. Duplicates with
consistent labels: drop one, no labeling work required.
Duplicates with conflicting labels: you must either relabel
or drop both, because even keeping one row means keeping a
lie in your training set.

---

### 6. Temporal and batch pattern analysis

Expert data scientists always check *when* duplicates were
created, not just *whether* they exist. A uniform low duplicate
rate distributed across the dataset's history suggests natural
noise. A spike in duplicate rate on a specific date, a specific
batch_id, or a specific data source suggests a concrete
upstream event (a double-ingestion, a reprocessing job that
ran twice, a pipeline that failed mid-flight and restarted
from the wrong checkpoint).

**What experts check here:**
- Duplicate rate as a function of timestamp or batch_id
  (requires a temporal column; the module should detect
  candidate temporal columns automatically if not specified)
- Concentration coefficient: what fraction of all duplicates
  come from the top-N% of time windows or batches? High
  concentration points to a specific incident; low
  concentration points to a structural pipeline bug
- Recency of duplicate spike: a spike in recent batches is
  more urgent than a historical spike, because it may still
  be propagating
- Whether the duplicate rate is monotonically increasing
  (growing technical debt) or episodic (a recoverable
  incident)

**Why root cause analysis depends on this:** "you have
duplicates" is a symptom. "You have duplicates because your
batch job double-ingested three specific pipeline runs in
March" is a diagnosis. The temporal analysis is what converts
the former into the latter.

---

### 7. Source and provenance analysis

Many real ML datasets are assembled from multiple upstream
sources, and duplicates often arise at join boundaries —
the same entity recorded in two source systems with
slightly different representations, merged without deduplication.
Expert data scientists check whether duplicates cluster by
data source when a source column is available.

**What experts check here:**
- Duplicate rate stratified by source column (if present)
- Cross-source duplicate rate: records that are duplicates
  of each other but come from different sources — almost
  always a data integration issue rather than a pipeline
  bug
- Within-source duplicate rate: records duplicated inside
  a single source — almost always a pipeline/ingestion
  issue specific to that source
- Whether cross-source duplicates have consistent or
  conflicting feature values for the same entity (consistent
  = safe to merge; conflicting = the two sources disagree
  about ground truth, which is a deeper problem than
  simple deduplication)

---

### 8. Train/validation/test split integrity

This is arguably the most important check in the whole
module for an ML diagnosis context specifically. Expert
data scientists check that the three splits are genuinely
independent at the entity level, not just at the row level.
A dataset can have zero exact row duplicates across splits
and still have catastrophic entity-level leakage if the
splitting was done naively on rows rather than on entities.

**What experts check here:**
- Entity overlap index: fraction of entities (by natural key)
  that appear in more than one split
- For each pair of splits (train-val, train-test, val-test):
  separate overlap rates, since train-test is more critical
  than train-val for evaluation validity
- Label distribution of overlapping entities: do overlapping
  entities have the same label distribution as non-overlapping
  ones? If overlapping entities have unusual label rates, the
  model has memorized exactly the hard cases, producing
  inflated metrics precisely where it matters most
- Random split vs. group split detection: given the overlap
  pattern, can we infer whether the dataset was split randomly
  on rows or by group/entity? A random row split will almost
  always produce entity overlap unless the dataset has unique
  entities per row by construction

**Root cause connection:** "validation AUC is 0.94 but
production AUC is 0.71" is one of the most common and most
painful ML failure modes. The first thing an expert checks
is entity-level train-test overlap. This check is the direct
diagnostic test for that failure mode.

---

### 9. Duplicate impact simulation

Once duplicates are identified, expert data scientists don't
just report them — they estimate the impact of removing them
before actually removing them. A dataset with 5% exact
duplicates loses 5% of its size after deduplication, but the
label distribution change, the feature coverage change, and
the effect on validation metrics all need to be understood
before deduplication is recommended.

**What experts check here:**
- Label distribution shift after deduplication: does removing
  duplicates disproportionately affect one class? A dataset
  where duplicates are concentrated in the minority class
  could become even more imbalanced after deduplication
- Size impact per split: how many rows are lost from each
  split, and does the resulting split size still support
  reliable model evaluation?
- Expected metric impact: given the label flip rate on
  conflicting duplicates, estimate the expected improvement
  in Bayes error after removing them vs. the expected
  decrease in training set size (the classic quality-quantity
  tradeoff)
- Whether deduplication would change the feature coverage
  (are duplicates concentrated in regions of feature space
  that would become underrepresented after removal?)

---

### 10. Recommended deduplication strategy

The final step an expert data scientist does is not just
"remove duplicates" but prescribe specifically how. The right
strategy depends entirely on what was found in steps 1–9.

**The four strategies experts choose between:**

**Drop-keep (deterministic):** when exact duplicates exist
within one split and labels are consistent — drop all but one.
Specify which one to keep (first-seen, last-seen, or highest
quality by a completeness score) and report how many rows
were affected per split.

**Merge (entity resolution):** when the same entity appears
with slightly different feature values across two source
systems — merge by taking the most complete version of each
field per entity. This is only safe when the analyst
understands which source is authoritative per field; the
module should flag when this decision requires human input.

**Group-aware re-split:** when entity-level overlap exists
across train/test because the original split was done naively
on rows — recommend re-splitting on entity groups rather
than patching the existing splits. This is the correct fix
for entity-level cross-split leakage; patching by removing
duplicates from test is the wrong fix because it introduces
selection bias.

**Flag-and-defer (conflicting labels):** when near-duplicates
have conflicting labels — do not silently drop either row.
Flag them as a relabeling task and remove them from training
until relabeled, because keeping either version of a
conflicting pair actively harms training. Report them as a
separate finding with the estimated Bayes error impact.

---

## Summary: what each check produces in Holmes terms

| Check | Collector Type | Signal Type | Causal Layer |
|---|---|---|---|
| Exact duplicate rate | ExactDuplicateCollector | ExactDuplicateRate | data |
| Exact cross-split overlap | SplitOverlapCollector | CrossSplitExactOverlap | data |
| Key uniqueness | KeyUniquenessCollector | KeyCollisionRate | data |
| Entity cross-split overlap | EntityOverlapCollector | EntityLeakage | data |
| Near-duplicate rate | FuzzyDuplicateCollector | NearDuplicateRate | data |
| Label flip rate | LabelConsistencyCollector | LabelFlipRate | labeling |
| Temporal duplicate spike | TemporalDuplicateCollector | DuplicateTemporalSpike | pipeline |
| Source-level duplicate rate | SourceProvenanceCollector | CrossSourceDuplicate | pipeline |
| Post-dedup label shift | ImpactSimulationCollector | DeduplicationLabelShift | data |

Note on causal_layer: temporal spike and cross-source findings
are tagged `pipeline` because they point to an upstream
collection/integration problem, not a characteristic of the
data itself. The distinction matters for root cause: a
`data` layer finding suggests the dataset needs repair; a
`pipeline` layer finding suggests the upstream system needs
repair and the dataset will self-heal once it does.

---

## What this module does NOT do

These are explicitly out of scope, stated clearly so the
module boundary is honest:

- **Entity resolution as a general-purpose product.** The
  module finds and characterizes duplicates for diagnostic
  purposes. It does not build a production deduplication
  pipeline. `simulate_retraining` via ISS is the right call
  if you want to quantify the impact of a deduplication
  strategy — not a built-in merge/dedupe engine.

- **Text similarity / semantic deduplication.** Near-duplicate
  detection here is structural (column values), not semantic
  (meaning). A record that says "NYC" and one that says
  "New York City" are not detected as near-duplicates by
  this module. That's a data cleaning problem, not a
  duplicate detection problem in the ML diagnostic sense.

- **Deduplication execution.** The module diagnoses and
  recommends; it does not modify the dataset. The
  recommendation output feeds a `Recommendation` object in
  `holmes.core`, which is then the user's decision to act on.