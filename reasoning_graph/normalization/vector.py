from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalizedVector:
    """
    The canonical numerical form of any evidence, regardless
    of its evidence_type. Contains exactly the four things the
    reasoning graph needs to do its math — nothing more.

    Every field has a fixed, consistent meaning across ALL
    evidence types. This is the contract the graph relies on.

    Fields:
        signal_strength:
            How notable/anomalous is this evidence?
            Always in [0, 1].
            0.0 = perfectly normal, nothing to see.
            1.0 = maximally anomalous/notable.

            This is what feeds into P(E|H) in belief propagation.
            A missing rate of 80% when baseline is 5% maps to ~0.95.
            A KS statistic with p=0.5 maps to ~0.1.

        reliability:
            How much should we trust this measurement method?
            Always in [0, 1].
            Copied from evidenceNode.reliability.
            A statistical test on 100k rows: ~0.95.
            A heuristic pattern match on 50 rows: ~0.30.

        polarity:
            Which direction does this push belief?
            +1.0 = this evidence supports anomaly / confirms a problem.
             0.0 = neutral / uninformative.
            -1.0 = this evidence suggests normality / rules out a problem.

            Explicit signed float rather than a string enum so the
            propagator can use it directly in weighted updates.

        coverage:
            What fraction of the subject space does this cover?
            Always in [0, 1].
            A whole-dataset evidence: 1.0.
            A single-feature evidence: feature_importance_weight (or 1/n_features).
            A segment evidence: segment_size / total_size.

            Used to scale how strongly this evidence propagates
            to neighboring nodes — a finding about one segment of
            100 rows shouldn't propagate as strongly as one about
            the entire 1M-row dataset.

    Source tracing (for clarity):
     evidence_id: links back to the full evidenceNode.
        normalization_record_id: links to the NormalizationRecord
            that explains exactly how each field was derived.
            This is the interpretability bridge.
    """
    signal_strength:           float   # [0, 1]
    reliability:               float   # [0, 1]
    polarity:                  float   # [-1, +1]
    coverage:                  float   # [0, 1]

    # Source tracing — never None, never optional
    evidence_id:            str
    normalization_record_id:   str

    def as_tuple(self) -> tuple[float, float, float, float]:
        """The four numbers the graph actually does math on."""
        return (self.signal_strength,
                self.reliability,
                self.polarity,
                self.coverage)
    
    


    def effective_weight(self) -> float:
        """
        The single scalar the propagator uses as a likelihood weight.
        Combines all four dimensions into one number, but the
        derivation is fully transparent and traceable.

        = signal_strength × reliability × |polarity| × sqrt(coverage)

        sqrt(coverage) rather than coverage directly — a segment
     evidence covering 10% of data shouldn't be penalized
        10x vs a global evidence, just ~3x.
        """
        import math
        return (self.signal_strength
                * self.reliability
                * abs(self.polarity)
                * math.sqrt(self.coverage))

    def __repr__(self):
        return (f"NormalizedVector("
                f"strength={self.signal_strength:.3f}, "
                f"reliability={self.reliability:.3f}, "
                f"polarity={self.polarity:+.3f}, "
                f"coverage={self.coverage:.3f}) "
                f"→ effective_weight={self.effective_weight():.3f}")