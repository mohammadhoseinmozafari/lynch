
from typing import List


from core.evidence.collector import EvidenceCollector
from core.evidence.evidence import Evidence
from core.evidence.evidence_type import EvidenceType
from core.observation.observation import Observation
from core.observation.type import ObservationType


class HighMissingRateEvidenceCollector (EvidenceCollector) :

    def __init__(self, normalizer) -> None:
        super().__init__(normalizer)
        self.evidence_type = EvidenceType.HIGH_MISSING_RATE
        self.supporting_type = ObservationType.COLUMN_MISSINGNESS


    def collect (self , observations : List[Observation]) -> List[Evidence]:
        observations = observations
        evidences: List[Evidence] = []
        for observation in observations :
                if observation.payload.get("missing_rate", 0.0) > 0.8:
                    evidences.append(self.build_evidence(observation))

        return evidences
    

    def build_evidence (self, observation : Observation) -> Evidence:
         
         return Evidence(
              evidence_type= self.evidence_type,
              source_observation_id= observation.id,
              normalized_vector_id= normalized.id,
              normalization_record_id= norm_record.id,
              
              collector_id=self.id,
         )
    

    


    