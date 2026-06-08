from enum import Enum
from typing import Dict, Optional

class Severity (Enum) :
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    
    @classmethod 
    def from_score (cls, score: float , thresholds : Optional[Dict] = None) -> 'Severity':
        """
        Convert a numerical score to a Severity level.
        
        Args:
            score: A float value to convert
            thresholds: Optional dict with keys 'medium', 'high', 'critical' 
                       specifying the minimum score for each level.
                       Default: {'medium': 4, 'high': 7, 'critical': 9}
            
        Returns:
            The corresponding Severity enum value
        """
        if thresholds is None:
            thresholds = {
                'medium': 4, 
                'high': 7, 
                'critical': 9
                }
        
        if score >= thresholds.get('critical', 9):
            return cls.CRITICAL
        
        elif score >= thresholds.get('high', 7):    
            return cls.HIGH
        
        elif score >= thresholds.get('medium', 4):
            return cls.MEDIUM
        
        else:
            return cls.LOW
    
    def __int__(self):
        """Allow conversion to integer."""
        return self.value