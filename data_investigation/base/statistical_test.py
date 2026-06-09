from abc import ABC, abstractmethod
from typing import List
from core.domain.entities.finding import Finding


class BaseStatisticalTest (ABC) :

    @abstractmethod
    def run(
        self,
        context
    ) -> List[Finding] :
        raise NotImplementedError