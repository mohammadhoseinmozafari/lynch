from  abc import ABC, abstractmethod
from typing import List
from core.domain.entities.finding import Finding

class BaseDetector (ABC):

    @abstractmethod
    def analyze (
        self,
        context,
    ) -> List [Finding]:
        
        
        raise NotImplementedError