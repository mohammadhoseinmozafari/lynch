from __future__ import annotations
from core.case.domain.value_objects import CodeSnapshot
from core.shared.domain.enums.rule_severity import RuleSeverity
from core.shared.domain.enums.warning_severity import WarningSeverity
from core.shared.domain.value_objects.validation_report import ValidationReport
from core.shared.domain.value_objects.validation_rule import ValidationRule
from core.shared.validators import BaseValidator
from core.case.domain.errors.code_snapshot_errors import CodeSnapshotErrorCode, CodeSnapshotWarningCode

import logging


logger = logging.getLogger(__name__)
class CodeSnapshotValidator(BaseValidator[CodeSnapshot]) :
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
        
        super().__init__(name = "CodeSnapshotValidator")
        self._verify_remote = verify_remote
        self._remote_timeout = remote_timeout_seconds

    
    def _register_rules(self) -> None:

        self.register_rule(
            ValidationRule(
                name="repository_validation",
                description="Validate Git repository URL format and special cases",
                severity=RuleSeverity.BLOCKING,
                dependencies=set(),
                func = self._validate_repository
            )
        )
        
        self.register_rule(ValidationRule(
                name="commit_validation",
                description="Validate commit hash format and zero-hash special case",
                severity=RuleSeverity.BLOCKING,
                dependencies={"repository_validation"},
                func=self._validate_commit
            )
        )
         # Rule 3: Entry point validation
        self.register_rule(ValidationRule(
            name="entry_point_validation",
            description="Validate entry point exists and check for notebooks",
            severity=RuleSeverity.BLOCKING,
            dependencies=set(),
            func=self._validate_entry_point
            )
        )

         # Rule 4: Diff validation
        self.register_rule(ValidationRule(
            name="diff_validation",
            description="Validate uncommitted diff format and size",
            severity=RuleSeverity.NON_BLOCKING,
            dependencies=set(),
            func=self._validate_diff
        ))

        # Rule 5: Remote commit verification (optional, depends on verify_remote flag)
        if self._verify_remote:
            self.register_rule(ValidationRule(
                name="remote_verification",
                description="Verify commit exists on remote repository",
                severity=RuleSeverity.NON_BLOCKING,
                dependencies={"repository_validation", "commit_validation"},
                func=self._verify_remote_commit
            ))
        else:
            # Add warning rule when remote verification is skipped
            self.register_rule(ValidationRule(
                name="remote_verification_skipped",
                description="Warning when remote verification is disabled",
                severity=RuleSeverity.NON_BLOCKING,
                dependencies={"repository_validation", "commit_validation"},
                func=self._add_remote_verification_warning
            ))



    def validate(self, target: CodeSnapshot, report: ValidationReport | None = None) -> ValidationReport:
        return super().validate(target, report)

    def _validate_repository (self, snapshot : CodeSnapshot,
                               report : ValidationReport, rule_name: str) -> None :
        if snapshot.git_repository == 'local' :
            report.add_warning(
                code = CodeSnapshotWarningCode.LOCAL_REPO_NO_GIT,
                detail = (
                    "No external Git repository. Reproducibility is limited. "
                    "This model cannot be deployed to production"
                ),
                severity = WarningSeverity.MEDIUM,
                context={
                    "repository": snapshot.git_repository, "rule": rule_name
                }
            )
            return
        if not any(
            snapshot.git_repository.startswith(pattern) for pattern in self.VALID_REPO_PATTERNS
        ):
            report.add_error(
                code = CodeSnapshotErrorCode.INVALID_REPO_URL,
                detail= (
                    f"Repository URL '{snapshot.git_repository}' doesn't match expected Git URL patterns : {','.join(self.VALID_REPO_PATTERNS)}"
                )
                ,
                context={"repository": snapshot.git_repository, "rule": rule_name}
            )
    
    def _validate_commit (self,
                            snapshot : CodeSnapshot ,
                            report : ValidationReport,
                            rule_name: str) -> None:
        """Validates the commit hash,
          pydantic ensures that the hash is a 40 or 7 hex string. 
          This validator checks if it's the zero placeholder
        """
        if snapshot.git_commit_hash == "0"*40 or snapshot.git_commit_hash == "0"*7:
            if snapshot.git_repository == 'local':
                pass
            else:
                report.add_error(
                    code= CodeSnapshotErrorCode.INVALID_COMMIT_HASH,
                    detail= (
                        "Zero commit hash is only allowed when git repository is local. Provide the actual commit hash."
                ),
                context={"commit_hash": snapshot.git_commit_hash, "rule": rule_name}
            )



    def _validate_entry_point(self,
                                snapshot: CodeSnapshot,
                                report : ValidationReport,
                                rule_name: str) -> None:
        """
        Validate the entry point field.
        
        """
        if snapshot.entry_point.endswith(self.NOTEBOOK_EXTENSIONS):
            report.add_warning(
                code=CodeSnapshotWarningCode.ENTRY_POINT_NOTEBOOK,
                detail=(
                    f"Entry point '{snapshot.entry_point}' is a Jupyter notebook. "
                    "Execution order is not guaranteed; cell outputs may be stale. "
                    "Consider exporting to a Python script for production reproducibility."
                ),
                severity=WarningSeverity.LOW,
                context={"entry_point": snapshot.entry_point, "rule": rule_name}
            )
    


    def _validate_diff(self,
                        snapshot: CodeSnapshot, 
                        report: ValidationReport,
                        rule_name: str) -> None:
        """Validate the uncommitted diff, if present."""
        if not snapshot.uncommitted_diff:
            return
        
        # Warn that uncommitted changes exist
        report.add_warning(
            code=CodeSnapshotWarningCode.UNCOMMITTED_CHANGES,
            detail=(
                "Uncommitted changes are present. The model may not be reproducible "
                "from the commit alone. Commit all changes for production deployments."
            ),
            severity=WarningSeverity.MEDIUM,
            context={"diff_size": len(snapshot.uncommitted_diff), "rule": rule_name}

        )
        
        # Check if diff was truncated
        if snapshot.uncommitted_diff.endswith("[TRUNCATED]"):
            report.add_warning(
                code=CodeSnapshotWarningCode.DIFF_TRUNCATED,
                detail=(
                    f"Uncommitted diff exceeded {self.MAX_DIFF_SIZE_BYTES:,} bytes "
                    "and was truncated. Some changes are not captured."
                ),
                severity=WarningSeverity.MEDIUM,
                context={"rule": rule_name}
            )
        
        # Basic check: does it look like a unified diff?
        if not self._looks_like_diff(snapshot.uncommitted_diff):
            report.add_warning(
                code=CodeSnapshotWarningCode.DIFF_NOT_PARSEABLE,
                detail=(
                    "The provided diff does not appear to be a valid unified diff. "
                    "It may not be usable for reproduction."
                ),
                severity=WarningSeverity.LOW,
                context={"rule": rule_name}
            )
    
    
    
    def _verify_remote_commit(self,
                               snapshot: CodeSnapshot, 
                               report: ValidationReport,
                               rule_name: str
    ) -> None:
        """
        Attempt to verify the commit exists on the remote repository.
        
        This performs network I/O and may fail due to:
          - Network issues
          - Authentication failures
          - Repository not found
          - Commit not found (force-pushed branch, deleted fork)
        """
        if snapshot.git_repository == 'local':
            return
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
                    code=CodeSnapshotWarningCode.REPO_NOT_REACHABLE,
                    detail=(
                        f"Could not reach repository '{snapshot.git_repository}': "
                        f"{result.stderr.strip()}"
                    ),
                    severity=WarningSeverity.MEDIUM,
                    context={"repository": snapshot.git_repository, "rule": rule_name}
                )
            elif snapshot.git_commit_hash not in result.stdout:
                report.add_warning(
                    code=CodeSnapshotWarningCode.COMMIT_NOT_FOUND,
                    detail=(
                        f"Commit '{snapshot.git_commit_hash[:8]}' not found in "
                        f"'{snapshot.git_repository}'. The branch may have been "
                        "force-pushed or the fork deleted."
                    ),
                    severity=WarningSeverity.MEDIUM,
                )
                
        except subprocess.TimeoutExpired:
            report.add_warning(
                code=CodeSnapshotWarningCode.REPO_NOT_REACHABLE,
                detail=(
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
    
    def _add_remote_verification_warning(self, snapshot: CodeSnapshot,
                                           report: ValidationReport,
                                           rule_name: str) -> None:
        """Add warning when remote verification is skipped."""
        if snapshot.git_repository != "local":
            report.add_warning(
                code=CodeSnapshotWarningCode.COMMIT_UNVERIFIED,
                detail=(
                    "Commit existence not verified against remote. "
                    "Lineage relies on client-provided data only."
                ),
                severity=WarningSeverity.MEDIUM,
                context={"rule": rule_name}
            )
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