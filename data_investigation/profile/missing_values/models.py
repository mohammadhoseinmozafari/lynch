
from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field

from typing import Any, Optional , Dict
from enum import Enum





@dataclass(frozen= True)
class FeatureMissingness  :
    """Raw missingness profile for a single feature."""

    feature_name : str 
    total_count : int 
    missing_count : int 
    missing_rate : MissingRate
    mechanism : Optional[MissingnessMechanism] = None
    target_correlation : Optional[float] = None




class MissingRate(BaseModel):
    """A validated missing rate in the interval [0.0, 1.0]."""
    value : float = Field (ge=0.0, le= 1.0) 
    model_config = ConfigDict(frozen= True)

    def is_critical (self) ->bool :
        return self.value >= 0.8
    
    def is_high (self) -> bool :
        return (self.value <0.8
                and 
                self.value >= 0.5
                )
    
    def is_moderate(self) -> bool :

        return (self.value < 0.5 
                and 
                self.value >=0.2
                )
    
    def is_low (self) -> bool:
        return self.value < 0.2
    

    def __repr__(self) -> str:
        return f"{self.value:.2%}"



class MissingnessMechanism (str , Enum):
    MCAR = 'MCAR'
    MAR = 'MAR'
    MNAR = 'MNAR'
    STRUCTURAL = 'STRUCTURAL'


@dataclass(frozen= True)
class MissingCluster : 
    """A sub‑population with disproportionately high missingness."""
    
    feature : str 
    segment : Dict[str, Any]
    segment_size : int
    missing_rate: MissingRate
    global_rate : float
    ratio : float 

