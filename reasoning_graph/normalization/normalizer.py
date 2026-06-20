from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.domain.entities.evidence import Evidence
from core.domain.enums.evidence_type import EvidenceType
from reasoning_graph.graph.nodes.nodes import EvidenceNode

from .vector import NormalizedVector
from .record import NormalizationRecord, FieldDerivation


class EvidenceNormalizer(ABC):
    """
    Base class for all normalizers. One normalizer per evidence_type,
    registered in the NormalizationEngine.

    The contract is strict:
    - Produce a NormalizedVector with exactly the four fields.
    - Produce a NormalizationRecord that fully explains each field.
    - Never raise an exception — return a degraded vector with
      low reliability if inputs are malformed (fail gracefully).
    - Never use information from outside the Evidence itself
      (no model state, no other evidences) — normalizers are
      pure functions of one evidence.
    """

    evidence_type: EvidenceType   # subclasses declare which type they handle

    @abstractmethod
    def normalize(self,
                  evidence : Evidence
                  ) -> tuple[NormalizedVector, NormalizationRecord]:
        """
        Produce the NormalizedVector and its NormalizationRecord.
        Both must be returned together — they are inseparable.
        """

    # ---- Protected helpers for subclasses ----
    # These are the building blocks normalizer authors use to
    # produce consistent, well-documented derivations.

    def _derive_strength_from_rate(self,
                                    value: float,
                                    baseline: float,
                                    rationale: str) -> FieldDerivation:
        """
        For RATE-shaped evidence: normalize the concentration ratio
        to [0,1] using a log scale (so 2x concentration gives a
        meaningfully different score from 20x, not a linear one).

        Formula: 1 - exp(-max(0, (value/baseline) - 1) / 3)
        """
        import math
        if baseline == 0:
            ratio = 1.0 if value > 0 else 0.0
        else:
            ratio = max(0.0, (value / baseline) - 1.0)
        raw = 1 - math.exp(-ratio / 3)
        return FieldDerivation(
            field_name="signal_strength",
            raw_input={"value": value, "baseline": baseline},
            formula="1 - exp(-max(0, (value/baseline) - 1) / 3)",
            formula_result=raw,
            final_value=min(1.0, max(0.0, raw)),
            rationale=rationale,
        )

    def _derive_strength_from_pvalue(self,
                                      p_value: float,
                                      rationale: str) -> FieldDerivation:
        """
        For TEST-shaped evidence: map a p-value to signal strength.
        Lower p-value = stronger signal.

        Formula: 1 - p_value^0.2
        (concave — makes small p-values discriminate better
         than a linear 1-p_value mapping)
        """
        raw = 1 - (p_value ** 0.2)
        return FieldDerivation(
            field_name="signal_strength",
            raw_input={"p_value": p_value},
            formula="1 - p_value^0.2",
            formula_result=raw,
            final_value=min(1.0, max(0.0, raw)),
            rationale=rationale,
        )

    def _derive_strength_from_percentile(self,
                                          score: float,
                                          score_at_95th: float,
                                          rationale: str) -> FieldDerivation:
        """
        For SCORE-shaped evidence: normalize a raw score using the
        known 95th-percentile reference point so the scale is
        interpretable across different scoring methods.

        Formula: min(1, score / score_at_95th)
        """
        raw = score / score_at_95th if score_at_95th > 0 else 0.0
        return FieldDerivation(
            field_name="signal_strength",
            raw_input={"score": score, "reference_95th": score_at_95th},
            formula="min(1, score / score_at_95th)",
            formula_result=raw,
            final_value=min(1.0, max(0.0, raw)),
            rationale=rationale,
        )

    def _derive_coverage_from_subject(self,
                                       subject: "Subject",
                                       dataset_size: int) -> FieldDerivation:
        """
        Derive coverage from the observation's subject:
        - Dataset level → 1.0
        - Feature level → 1.0 (it covers the whole feature)
        - Segment level → segment_size / dataset_size
        - Row level     → len(row_ids) / dataset_size
        """
        if subject.level in ("dataset", "feature", "model", "pipeline"):
            coverage = 1.0
            raw_input = {"subject_level": subject.level}
            formula = "1.0 (full-scope subject)"
        elif subject.segment_filter is not None:
            seg_size = subject.metadata.get("segment_size", dataset_size)
            coverage = min(1.0, seg_size / dataset_size) if dataset_size > 0 else 0.5
            raw_input = {"segment_size": seg_size, "dataset_size": dataset_size}
            formula = "segment_size / dataset_size"
        elif subject.row_ids is not None:
            coverage = min(1.0, len(subject.row_ids) / dataset_size)
            raw_input = {"n_rows": len(subject.row_ids), "dataset_size": dataset_size}
            formula = "len(row_ids) / dataset_size"
        else:
            coverage = 1.0
            raw_input = {}
            formula = "1.0 (default)"

        return FieldDerivation(
            field_name="coverage",
            raw_input=raw_input,
            formula=formula,
            formula_result=coverage,
            final_value=coverage,
            rationale=f"Subject level is '{subject.level}'"
        )

    def _make_record(self,
                     observation_id: str,
                     derivations: list[FieldDerivation]) -> NormalizationRecord:
        import uuid
        return NormalizationRecord(
            id=str(uuid.uuid4()),
            observation_id=observation_id,
            evidence_type=self.evidence_type,
            normalizer_id=self.__class__.__name__,
            field_derivations=derivations,
        )

    def _fallback(self,
                  observation: "ObservationNode",
                  reason: str) -> tuple[NormalizedVector, NormalizationRecord]:
        """
        Graceful degradation when normalization fails.
        Produces a low-reliability, neutral vector rather than crashing.
        """
        import uuid
        vec = NormalizedVector(
            signal_strength=0.0,
            reliability=0.1,
            polarity=0.0,
            coverage=0.0,
            observation_id=observation.id,
            normalization_record_id="fallback",
        )
        record = NormalizationRecord(
            id=str(uuid.uuid4()),
            observation_id=observation.id,
            evidence_type=observation.evidence_type,
            normalizer_id=self.__class__.__name__,
            field_derivations=[
                FieldDerivation(
                    field_name="signal_strength",
                    raw_input={},
                    formula="fallback",
                    formula_result=0.0,
                    final_value=0.0,
                    rationale=f"Normalization failed: {reason}",
                )
            ]
        )
        return vec, record