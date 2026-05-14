from enum import Enum

class CodeSnapshotErrorCode(str, Enum):
    """Machine-readable error codes for code snapshot validation failures."""
    INVALID_REPO_URL = "INVALID_REPO_URL"
    INVALID_COMMIT_HASH = "INVALID_COMMIT_HASH"
    MISSING_ENTRY_POINT = "MISSING_ENTRY_POINT"


class CodeSnapshotWarningCode(str, Enum):
    """Machine-readable warning codes for code snapshot validation."""
    LOCAL_REPO_NO_GIT = "LOCAL_REPO_NO_GIT"
    UNCOMMITTED_CHANGES = "UNCOMMITTED_CHANGES"
    BRANCH_NOT_TAG = "BRANCH_NOT_TAG"
    DIFF_TRUNCATED = "DIFF_TRUNCATED"
    DIFF_NOT_PARSEABLE = "DIFF_NOT_PARSEABLE"
    REPO_NOT_REACHABLE = "REPO_NOT_REACHABLE"
    COMMIT_NOT_FOUND = "COMMIT_NOT_FOUND"
    COMMIT_UNVERIFIED = "COMMIT_UNVERIFIED"
    ENTRY_POINT_NOTEBOOK = "ENTRY_POINT_NOTEBOOK"