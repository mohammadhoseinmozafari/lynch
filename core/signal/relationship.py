from .base import Signal
from .enums import SignalCategory


class RelationshipSignal(Signal):

    category = SignalCategory.RELATIONSHIP

    source_subject: str

    target_subject: str

    relation_type: str

    association_strength: float