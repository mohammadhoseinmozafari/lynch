from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthScore(BaseModel) :
    """
    A 0–100 health score with optional dimensional breakdown and trend.

    The overall score is the primary indicator of health. The breakdown
    shows how individual dimensions (e.g., "missing_values", "outliers",
    "leakage") contribute to the overall score. The trend shows whether
    the score is improving or deteriorating compared to a previous report.

    Attributes:
        overall: The overall health score, 0.0–100.0. Higher is better.
        breakdown: Per-dimension scores keyed by dimension name.
            Each dimension also ranges 0–100.
        trend: The change in overall score since the last report.
            Positive means improving, negative means declining.
            None if this is the first report.

    Example:
        >>> score = HealthScore(
        ...     overall=78.5,
        ...     breakdown={"missing_values": 65.0, "outliers": 90.0, "leakage": 85.0},
        ...     trend=+3.2
        ... )
        >>> score.grade()
        'B'
        >>> score.is_healthy()
        True
        >>> score.is_healthy(threshold=80.0)
        False
    """

    overall : float = Field(default= 100.0, ge = 0.0 , le= 100.0)
    breakdown : Dict[str, float] = Field (default_factory= dict)
    trend : Optional[float] = Field (default= None)

    def grade(self) -> str:
        """
        Return a letter grade for the overall score.

        Grading scale:
            A:  90.0 – 100.0  (Excellent)
            B:  80.0 – 89.9   (Good)
            C:  70.0 – 79.9   (Fair)
            D:  50.0 – 69.9   (Poor)
            F:   0.0 – 49.9   (Critical)

        Returns:
            A single letter grade string.
        """
        if self.overall >= 90.0:
            return "A"
        elif self.overall >= 80.0:
            return "B"
        elif self.overall >= 70.0:
            return "C"
        elif self.overall >= 50.0:
            return "D"
        else:
            return "F"

    def is_healthy(self, threshold: float = 70.0) -> bool:
        """
        Check whether the overall score meets or exceeds a threshold.

        Args:
            threshold: Minimum score to be considered healthy.
                Default is 70.0 (grade C or better).

        Returns:
            True if overall >= threshold, False otherwise.
        """
        return self.overall >= threshold
