from __future__ import annotations
from typing import Optional
from pydantic import BaseModel



class Metrics (BaseModel) :
    """
    Captures the evaluation metrics of the training run,
    organised by dataset split. This is stored inside 
    the TrainingRun and used for explanation‑quality gating,
    deployment validation, and cryptographic lineage.

    Attributes:
        metrics: Mapping from split name ("train", "validation", "test", etc.)
          to the metrics computed on that split. At least "validation" should be present unless no validation was performed (warning raised).
        primary_metric: Name of the primary evaluation metric, e.g., "accuracy", "f1".
          Should match a key inside one of the MetricGroup dictionaries. Used for deployment readiness checks.
    """
    metrics : dict[str, MetricGroup]
    primary_metric : Optional[str] = None


class MetricGroup (BaseModel) :
    """
    A collection of metrics for a single dataset split
    Attributes:
        metrics: Mapping from metric name (non‑empty, e.g., "accuracy", "rmse", "log_loss") to its computed value.
          At least one metric per split is required.
    """
    metrics : dict[str, MetricValue]


class MetricValue (BaseModel) :
    """
    A single evaluation metric value,
      optionally accompanied by a confidence interval.
    Attributes:
        value: The scalar metric value. Must be a finite number (NaN and Inf are rejected during validation).
        confidence_interval: If the metric has an associated confidence interval (e.g., from bootstrapping or multiple folds).
        step: For models that log metrics at intermediate steps (epochs, batches). Kept as an optional informational field; not used for lineage.
    """
    value : float
    confidence_interval : Optional[ConfidenceInterval] = None
    step : Optional[int]


class ConfidenceInterval (BaseModel):
    lower : float
    upper : float
    confidence_level : Optional[float] = 0.95