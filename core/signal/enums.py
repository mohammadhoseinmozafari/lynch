
from enum import Enum

class SignalCategory(Enum , str) :
    HEALTH = "health"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    RELATIONSHIP = "relationship"
    DIAGNOSTIC = "diagnostic"
    CONFIDENCE = "confidence"



class SubjectType(str, Enum):
    DATASET = "dataset"
    FEATURE = "feature"
    ROW = "row"
    MODEL = "model"
    PIPELINE = "pipeline"
    SYSTEM = "system"