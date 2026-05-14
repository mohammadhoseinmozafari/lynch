
from pydantic import BaseModel
from typing import Optional , List

class NumericRange (BaseModel):
    """
    Numeric range is a simple range (min,max). 
    """
    min : Optional[float] = None
    max : Optional[float] = None

    

class CategoricalRange (BaseModel) :
    """
    For categorical features we cannot define a numerical range. 
    The categorical range is the possible values of a categorical feature.
    """
    possible_values : List[str]
