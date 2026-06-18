from abc import ABC


class PatternRecognizer(ABC):

    @property
    def name(self):
        pass

    @property
    def produces(self):
        pass

    @property
    def consumes(self):
        """
        observation types

        OR

        pattern types
        """
        pass
    
    def recognize(
        self,
        observations,
        patterns
    )-> list[PatternInstance]:

        pass