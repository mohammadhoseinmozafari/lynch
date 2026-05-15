from enum import Enum


class ValidatorExecutionStatus(str, Enum):
    """Status of a validator's execution."""
    SUCCESS = "success"           # Completed without errors
    FAILED = "failed"             # Completed but found validation errors
    CRASHED = "crashed"           # Unexpected exception during execution
    SKIPPED = "skipped"           # Skipped due to fail-fast or extraction failure
    NOT_RUN = "not_run"           # Not executed yet