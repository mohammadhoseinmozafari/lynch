

from core.domain.entities.evidence import Evidence
from core.domain.enums.evidence_type import EvidenceType
from core.normalization.normalizer import EvidenceNormalizer


class ColumnMissingRateNormalizer (EvidenceNormalizer):
    evidence_type = EvidenceType.COLUMN_MISSINGNESS

    def normalize   (
            self,
            evidence : Evidence
    ):
        
        