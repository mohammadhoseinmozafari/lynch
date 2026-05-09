from __future__ import annotations
import hashlib
import logging
from core.case.code import CodeSnapshot
from core.case.validation import CodeSnapshotValidation, CodeCompleteness
from core.case.validation.code_snapshot_validator import CodeSnapshotErrorCode, CodeSnapshotWarningCode, WarningSeverity
logger = logging.getLogger(__name__)
class CodeSnapshotValidator :
    """
    Validates a CodeSnapshot for reproducibility and explainability trust.
    
    This is a CUSTOM validator. Pydantic already validated:
      - git_commit_hash is a 40-char hex string
      - git_repository is a non-empty string
      - entry_point is a non-empty string
      - uncommitted_diff is a string (if provided) within 1MB
    
    This validator checks DOMAIN rules:
      - Is the repo URL actually valid?
      - Does the commit exist on the remote?
      - Is the diff parseable as a unified diff?
      - What is the overall reproducibility classification?
    """

    # Maximum uncommitted diff size (1mb)
    MAX_DIFF_SIZE_BYTES = 1_000_000
    VALID_REPO_PATTERNS = ("https://", "git@", "ssh://", "http://")
    NOTEBOOK_EXTENSIONS = (".ipynb")

    def __init__(self, 
                 verify_remote : bool = False,
                 remote_timeout_seconds : int =10
                 ) -> None:
        """
        Args:
            verify_remote: if True, attempt to connect to the remote repo (requires network I/O for validation)
            remote_timeout_seconds: Timeout for remote verification.
        """
        self._verify_remote = verify_remote
        self._remote_timeout = remote_timeout_seconds

    def validate (self, snapshot: CodeSnapshot) -> CodeSnapshotValidation :
        """
        Validate a code snapshot for reproducibility
        Args: 
            snapshot: The CodeSnapshot to validate 
        Returns:
            A CodeSnapshotValidation report containing errors and warnings

        """
    
        report = CodeSnapshotValidation(is_valid=True)

        self._validate_repository(snapshot, report)

        self._validate_commit (snapshot, report)

        self._validate_entry_point(snapshot, report)
        
        self._validate_diff(snapshot, report)

        if self._verify_remote and snapshot.git_repository != "local":
            self._verify_remote_commit(snapshot, report)
        elif snapshot.git_repository != "local":
            report.add_warning(
                field="git_commit_hash",
                code=CodeSnapshotWarningCode.COMMIT_UNVERIFIED,
                message=(
                    "Commit existence not verified against remote. "
                    "Lineage relies on client-provided data only."
                ),
                severity=WarningSeverity.MEDIUM,
            )
        
        # 6. Determine completeness classification
        report.completeness = self._classify(snapshot, report)
        
        # 7. Compute hash if valid
        if report.is_valid:
            report.computed_hash = self._compute_hash(snapshot)
        
        return report

    def _validate_repository (self, snapshot : CodeSnapshot,
                               report : CodeSnapshotValidation) -> None :
        if snapshot.git_repository == 'local' :
            report.add_warning(
                field = 'git_repository',
                code = CodeSnapshotWarningCode.LOCAL_REPO_NO_GIT,
                message = (
                    "No external Git repository. Reproducibility is limited. "
                    "This model cannot be deployed to production"
                ),
                severity = WarningSeverity.MEDIUM
            )
            return
        if not any(
            snapshot.git_repository.startswith(pattern) for pattern in self.VALID_REPO_PATTERNS
        ):
            report.add_error(
                field = "git_repository",
                code = CodeSnapshotErrorCode.INVALID_REPO_URL,
                message = (
                    f"Repository URL '{snapshot.git_repository}' doesn't match expected Git URL patterns : {','.join(self.VALID_REPO_PATTERNS)}"
                )
            )
    def _validate_commit (self,
                            snapshot : CodeSnapshot ,
                            report : CodeSnapshotValidation) -> None:
        """Validates the commit hash,
          pydantic ensures that the hash is a 40 or 7 hex string. 
          This validator checks if it's the zero placeholder
        """
        if snapshot.git_commit_hash == "0"*40 or snapshot.git_commit_hash == "0"*7:
            if snapshot.git_repository == 'local':
                pass
            else:
                report.add_error(
                    field= "git_commit_hash",
                    code= CodeSnapshotErrorCode.INVALID_COMMIT_HASH,
                    message= (
                        "Zero commit hash is only allowed when git repository is local. Provide the actual commit hash."
                )
            )

    def _validate_entry_point(self,
                                snapshot: CodeSnapshot,
                                report : CodeSnapshotValidation) -> None:
        """
        Validate the entry point field.
        
        """
        if not snapshot.entry_point or not snapshot.entry_point.strip():
            report.add_error(
                field = "entry_point",
                code = CodeSnapshotErrorCode.MISSING_ENTRY_POINT,
                message = "Entry point (training script or notebook path) is required."
            )
            return
        if snapshot.entry_point.endswith(self.NOTEBOOK_EXTENSIONS):
            report.add_warning(
                field="entry_point",
                code=CodeSnapshotWarningCode.ENTRY_POINT_NOTEBOOK,
                message=(
                    f"Entry point '{snapshot.entry_point}' is a Jupyter notebook. "
                    "Execution order is not guaranteed; cell outputs may be stale. "
                    "Consider exporting to a Python script for production reproducibility."
                ),
                severity=WarningSeverity.LOW,
            )
    
    def _validate_diff(self, snapshot: CodeSnapshot, report: CodeSnapshotValidation) -> None:
        """Validate the uncommitted diff, if present."""
        if not snapshot.uncommitted_diff:
            return
        
        # Warn that uncommitted changes exist
        report.add_warning(
            field="uncommitted_diff",
            code=CodeSnapshotWarningCode.UNCOMMITTED_CHANGES,
            message=(
                "Uncommitted changes are present. The model may not be reproducible "
                "from the commit alone. Commit all changes for production deployments."
            ),
            severity=WarningSeverity.MEDIUM,
        )
        
        # Check if diff was truncated
        if snapshot.uncommitted_diff.endswith("[TRUNCATED]"):
            report.add_warning(
                field="uncommitted_diff",
                code=CodeSnapshotWarningCode.DIFF_TRUNCATED,
                message=(
                    f"Uncommitted diff exceeded {self.MAX_DIFF_SIZE_BYTES:,} bytes "
                    "and was truncated. Some changes are not captured."
                ),
                severity=WarningSeverity.MEDIUM,
            )
        
        # Basic check: does it look like a unified diff?
        if not self._looks_like_diff(snapshot.uncommitted_diff):
            report.add_warning(
                field="uncommitted_diff",
                code=CodeSnapshotWarningCode.DIFF_NOT_PARSEABLE,
                message=(
                    "The provided diff does not appear to be a valid unified diff. "
                    "It may not be usable for reproduction."
                ),
                severity=WarningSeverity.LOW,
            )
    
    def _verify_remote_commit(
        self, snapshot: CodeSnapshot, report: CodeSnapshotValidation
    ) -> None:
        """
        Attempt to verify the commit exists on the remote repository.
        
        This performs network I/O and may fail due to:
          - Network issues
          - Authentication failures
          - Repository not found
          - Commit not found (force-pushed branch, deleted fork)
        """
        import subprocess
        
        try:
            result = subprocess.run(
                ["git", "ls-remote", snapshot.git_repository, snapshot.git_commit_hash],
                capture_output=True,
                text=True,
                timeout=self._remote_timeout,
            )
            
            if result.returncode != 0:
                report.add_warning(
                    field="git_repository",
                    code=CodeSnapshotWarningCode.REPO_NOT_REACHABLE,
                    message=(
                        f"Could not reach repository '{snapshot.git_repository}': "
                        f"{result.stderr.strip()}"
                    ),
                    severity=WarningSeverity.MEDIUM,
                )
            elif snapshot.git_commit_hash not in result.stdout:
                report.add_warning(
                    field="git_commit_hash",
                    code=CodeSnapshotWarningCode.COMMIT_NOT_FOUND,
                    message=(
                        f"Commit '{snapshot.git_commit_hash[:8]}' not found in "
                        f"'{snapshot.git_repository}'. The branch may have been "
                        "force-pushed or the fork deleted."
                    ),
                    severity=WarningSeverity.MEDIUM,
                )
                
        except subprocess.TimeoutExpired:
            report.add_warning(
                field="git_repository",
                code=CodeSnapshotWarningCode.REPO_NOT_REACHABLE,
                message=(
                    f"Timed out connecting to '{snapshot.git_repository}' "
                    f"after {self._remote_timeout}s."
                ),
                severity=WarningSeverity.MEDIUM,
            )
        except FileNotFoundError:
            # git not installed
            logger.warning("git command not found; skipping remote verification")
        except Exception as exc:
            logger.error(f"Unexpected error during remote verification: {exc}")
    
    def _classify(self,
                    snapshot: CodeSnapshot,
                    report: CodeSnapshotValidation) -> CodeCompleteness:
            """
            Determine the reproducibility classification.
        
            Rules:
            - COMPLETE:   Valid git repo, full commit, no uncommitted diff, entry point provided.
            - PARTIAL:    Valid repo and commit, but uncommitted diff present or branch is dirty.
            - MINIMAL:    git_repository="local" or zero commit hash.
            - UNKNOWN:    Remote verification failed and no local data to fall back on.
            """
            if not report.is_valid:
                return CodeCompleteness.UNKNOWN
            
            if snapshot.git_repository == "local" or snapshot.git_commit_hash == "0" * 40:
                return CodeCompleteness.MINIMAL
            
            has_diff = snapshot.uncommitted_diff is not None and len(snapshot.uncommitted_diff) > 0
            has_verified_remote = not any(
                w.code == CodeSnapshotWarningCode.COMMIT_UNVERIFIED
                for w in report.warnings
            )
            
            if has_diff:
                return CodeCompleteness.PARTIAL
            
            if not has_verified_remote:
                # Could not verify remote but client provided valid-looking data
                return CodeCompleteness.PARTIAL
            
            return CodeCompleteness.COMPLETE
    
    # -------------------------------------------------------------------------
    # Hashing
    # -------------------------------------------------------------------------
    
    def _compute_hash(self, snapshot: CodeSnapshot) -> str:
        """
        Compute a deterministic SHA256 hash of the code snapshot.
        
        The hash includes all fields that affect reproducibility.
        """
        import json
        
        payload = json.dumps(
            {
                "git_repository": snapshot.git_repository,
                "git_commit_hash": snapshot.git_commit_hash,
                "entry_point": snapshot.entry_point,
                "uncommitted_diff": snapshot.uncommitted_diff or "",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _looks_like_diff(text: str) -> bool:
        """
        Heuristic check: does this text look like a unified diff?
        
        A unified diff typically starts with lines like:
          diff --git a/... b/...
          --- a/...
          +++ b/...
          @@ -x,y +x,y @@
        """
        indicators = [
            "diff --git ",
            "--- a/",
            "+++ b/",
            "@@ -",
        ]
        return any(indicator in text for indicator in indicators)