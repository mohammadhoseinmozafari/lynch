from __future__ import annotations
from typing import Optional, Union, List, Dict
from pydantic import BaseModel



class HyperparameterBundle (BaseModel) :
    parameters : dict [str, HyperparameterValue]
    framework_version : Optional[str] = None
    notes : Optional[str] = None


HyperparameterValue = Union [
    int,
    float,
    bool,
    str,
    List['HyperparameterValue'],
    Dict[str, 'HyperparameterValue']
]

