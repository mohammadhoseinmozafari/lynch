import json
import re
from typing import Optional
from packaging.version import Version, InvalidVersion
from packaging.requirements import Requirement
from core.case.domain.errors.env_snapshot_errors import EnvironmentSnapshotErrorCode, EnvironmentSnapshotWarningCode
from core.case.domain.value_objects.environment_snapshot import EnvironmentSnapshot
from core.case.domain.enums.env_snapshot_type import EnvSnapshotType

from core.shared.domain.enums.rule_severity import RuleSeverity
from core.shared.domain.enums.warning_severity import WarningSeverity
from core.shared.domain.value_objects.validation_report import ValidationReport
from core.shared.domain.value_objects.validation_rule import ValidationRule
from core.shared.validators import BaseValidator
   



class EnvironmentValidator(BaseValidator[EnvironmentSnapshot]):
    """
    Validator for EnvironmentSnapshot objects.
    Ensures all dependencies are pinned to exact versions and specifications are parseable.
    """
    
    def __init__(self):
        super().__init__(name="EnvironmentValidator")
    
    def _get_target_id(self, target: EnvironmentSnapshot) -> Optional[str]:
        """Extract unique identifier from environment snapshot."""
        return f"{target.snapshot_type.value}:{target.python_version}"
    
    def _register_rules(self) -> None:
        """Register all validation rules for environment snapshots."""
        
        # ====================================================================
        # BLOCKING Rules - Must pass before dependent rules can run
        # ====================================================================
        self.register_rule(ValidationRule(
            name="validate_python_version_format",
            description="Check that python_version is a valid, pinned version",
            severity=RuleSeverity.BLOCKING,
            func=self._validate_python_version_format,
            dependencies=set()
        ))
        
        # ====================================================================
        # Format-specific BLOCKING Rules
        # ====================================================================
        
        self.register_rule(ValidationRule(
            name="validate_pip_lock_format",
            description="Validate PIP lock file format and pinning",
            severity=RuleSeverity.BLOCKING,
            func=self._validate_pip_lock_format,
            dependencies=set()
        ))
        
        self.register_rule(ValidationRule(
            name="validate_conda_lock_format",
            description="Validate conda-lock JSON format and pinning",
            severity=RuleSeverity.BLOCKING,
            func=self._validate_conda_lock_format,
            dependencies=set()
        ))
    
        
        # ====================================================================
        # NON-BLOCKING Rules - Warnings and additional checks
        # ====================================================================
        
        self.register_rule(ValidationRule(
            name="check_system_packages_pinned",
            description="Check that system packages are pinned to exact versions",
            severity=RuleSeverity.NON_BLOCKING,
            func=self._check_system_packages_pinned,
            dependencies=set()
        ))
        
        self.register_rule(ValidationRule(
            name="check_python_version_stability",
            description="Warn if Python version is pre-release or development",
            severity=RuleSeverity.NON_BLOCKING,
            func=self._check_python_version_stability,
            dependencies={"validate_python_version_format"}
        ))
        
        self.register_rule(ValidationRule(
            name="check_dependency_hashes",
            description="Warn if dependencies don't include hashes for integrity",
            severity=RuleSeverity.NON_BLOCKING,
            func=self._check_dependency_hashes,
            dependencies={
                "validate_pip_lock_format",
                "validate_conda_lock_format"
            }
        ))
        
        self.register_rule(ValidationRule(
            name="check_redundant_dependencies",
            description="Warn about potentially redundant or conflicting dependencies",
            severity=RuleSeverity.NON_BLOCKING,
            func=self._check_redundant_dependencies,
            dependencies={
                "validate_pip_lock_format",
                "validate_conda_lock_format"
            }
        ))
    
    # ========================================================================
    # BLOCKING Validation Rule Functions
    # ========================================================================   
    
    def _validate_python_version_format(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Validate Python version is properly formatted and pinned."""
        version_str = target.python_version
        
        # Check for version range indicators
        if any(char in version_str for char in ['>', '<', '~', '!', '*', ',', '||']):
            report.add_error(
                code=EnvironmentSnapshotErrorCode.UNPINNED_PYTHON_VERSION,
                detail=f"Python version must be pinned to an exact version, "
                       f"got range specifier: '{version_str}'",
                context={"python_version": version_str}
            )
            return
        
        # Try to parse as a version
        try:
            version = Version(version_str)
            
            # Check for development versions
            if version.is_devrelease:
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.DEV_PYTHON_VERSION,
                    detail=f"Warning!,You are using development Python versions: {version_str}",
                    context={"python_version": version_str},
                    severity=WarningSeverity.HIGH
                )
            
            # Check for local versions
            if version.local:
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.LOCAL_PYTHON_VERSION,
                    detail=f"Warning! You are using local Python versions: {version_str}",
                    context={"python_version": version_str},
                    severity=WarningSeverity.HIGH
                )
                
        except InvalidVersion:
            report.add_error(
                code=EnvironmentSnapshotErrorCode.INVALID_PYTHON_VERSION,
                detail=f"Invalid Python version format: '{version_str}'. "
                       f"Expected format like '3.10.12', '3.10' or '3'",
                context={"python_version": version_str}
            )
    
    # ========================================================================
    # Format-Specific BLOCKING Validation Functions
    # ========================================================================
    
    def _validate_pip_lock_format(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Validate PIP lock format - only runs for PIP_LOCK type."""
        if target.snapshot_type != EnvSnapshotType.PIP_LOCK:
            return
        
        lines = target.specification.strip().split('\n')
        has_dependencies = False
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip pip options (--index-url, etc.)
            if line.startswith('-'):
                continue
            
            has_dependencies = True
            
            # Try to parse as a requirement
            try:
                req = Requirement(line)
                
                # Check for version pinning
                if not req.specifier:
                    report.add_error(
                        code=EnvironmentSnapshotErrorCode.MISSING_VERSION_PIN,
                        detail=f"Line {line_num}: Package '{req.name}' has no version pinned",
                        context={"line": line_num, "package": req.name}
                    )
                    
                
                # Check for exact version pinning
                has_exact_pin = False
                specifier_str = str(req.specifier)
                
                for spec in req.specifier:
                    if spec.operator == '==':
                        has_exact_pin = True
                        break
                
                if not has_exact_pin:
                    report.add_error(
                        code=EnvironmentSnapshotErrorCode.UNEXACT_VERSION_PIN,
                        detail=f"Line {line_num}: Package '{req.name}' must be pinned with '==', "
                               f"got: {specifier_str}",
                        context={
                            "line": line_num, 
                            "package": req.name,
                            "specifier": specifier_str
                        }
                    )
                
                # Check for extras (not allowed in lock files)
                if req.extras:
                    report.add_warning(
                        code=EnvironmentSnapshotWarningCode.EXTRAS_IN_LOCK_FILE,
                        detail=f"Line {line_num}: Package extras are not recommended in lock file: "
                               f"'{req.name}[{','.join(req.extras)}]'",
                        severity=WarningSeverity.HIGH
                        ,context={
                            "line": line_num,
                            "package": req.name,
                            "extras": list(req.extras)
                        }
                    )
                
                # Check for environment markers (may indicate non-deterministic behavior)
                if req.marker:
                    report.add_warning(
                        code= EnvironmentSnapshotWarningCode.MARKERS_IN_LOCK_FILE,
                        detail=f"Line {line_num}: Environment markers found in lock file "
                               f"for '{req.name}': {req.marker}",
                        severity=WarningSeverity.MEDIUM,
                        context={
                            "line": line_num,
                            "package": req.name,
                            "marker": str(req.marker)
                        }
                    )
                
                # Check for URL-based dependencies (may not be pinned)
                if req.url:
                    report.add_warning(
                        code=EnvironmentSnapshotWarningCode.URL_DEPENDENCY,
                        detail=f"Line {line_num}: URL-based dependency for '{req.name}' "
                               f"may not be properly versioned: {req.url}",
                        severity=WarningSeverity.MEDIUM,
                        context={
                            "line": line_num,
                            "package": req.name,
                            "url": req.url
                        }
                    )
                
            except Exception as e:
                report.add_error(
                    code= EnvironmentSnapshotErrorCode.INVALID_PIP_REQUIREMENT,
                    detail=f"Line {line_num}: Invalid requirement format: '{line}' - {str(e)}",
                    context={"line": line_num, "requirement": line, "error": str(e)}
                )
        
        if not has_dependencies:
            report.add_warning(
                code = EnvironmentSnapshotWarningCode.NO_DEPENDENCIES_FOUND,
                detail="No dependency requirements found in PIP lock specification",
                severity=WarningSeverity.HIGH,
                context={"specification": target.specification[:100]}
            )
    
    def _validate_conda_lock_format(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Validate conda-lock format - only runs for CONDA_LOCK type."""
        if target.snapshot_type != EnvSnapshotType.CONDA_LOCK:
            return
        
        try:
            data = json.loads(target.specification)
        except json.JSONDecodeError as e:
            report.add_error(
                code=EnvironmentSnapshotErrorCode.INVALID_JSON_FORMAT,
                detail=f"Invalid JSON in conda-lock specification: {str(e)}",
                context={"error_line": e.lineno, "error_column": e.colno}
            )
            return
        
        # Check for required conda-lock structure
        if not isinstance(data, dict):
            report.add_error(
                code="INVALID_CONDA_LOCK_STRUCTURE",
                detail="Conda-lock specification must be a JSON object",
                context={"type": type(data).__name__}
            )
            return
        
        if 'version' not in data:
            report.add_error(
                code=EnvironmentSnapshotErrorCode.MISSING_CONDA_LOCK_VERSION,
                detail="Conda-lock specification missing required 'version' field",
                context={"keys": list(data.keys())}
            )
        
        # Validate packages
        packages = data.get('package', [])
        if not packages:
            report.add_error(
                code=EnvironmentSnapshotErrorCode.NO_PACKAGES_IN_CONDA_LOCK,
                detail="No packages found in conda-lock specification",
                context={"has_package_key": 'package' in data}
            )
            return
        
        if not isinstance(packages, list):
            report.add_error(
                code=EnvironmentSnapshotErrorCode.INVALID_PACKAGES_FORMAT,
                detail="'package' field must be a list",
                context={"type": type(packages).__name__}
            )
            return
        
        # Validate each package
        for idx, pkg in enumerate(packages):
            if not isinstance(pkg, dict):
                report.add_error(
                    code=EnvironmentSnapshotErrorCode.INVALID_PACKAGE_ENTRY,
                    detail=f"Package at index {idx} must be an object, got {type(pkg).__name__}",
                    context={"index": idx, "type": type(pkg).__name__}
                )
                continue
            
            pkg_name = pkg.get('name', f'<unknown-{idx}>')
            
            # Check required fields
            if 'name' not in pkg:
                report.add_error(
                    code=EnvironmentSnapshotErrorCode.MISSING_PACKAGE_NAME,
                    detail=f"Package at index {idx} missing 'name' field",
                    context={"index": idx}
                )
            
            if 'version' not in pkg:
                report.add_error(
                    code=EnvironmentSnapshotErrorCode.MISSING_PACKAGE_VERSION,
                    detail=f"Package '{pkg_name}' missing 'version' field",
                    context={"package": pkg_name}
                )
            else:
                # Check version is pinned
                version = pkg['version']
                if any(char in version for char in ['>', '<', '~', '*', '|', ',']):
                    report.add_error(
                        code=EnvironmentSnapshotErrorCode.UNPINNED_CONDA_PACKAGE,
                        detail=f"Package '{pkg_name}' must be pinned to exact version, "
                               f"got: '{version}'",
                        context={"package": pkg_name, "version": version}
                    )
            
            # Validate URL if present
            if 'url' in pkg and isinstance(pkg['url'], str):
                url = pkg['url']
                if not url.startswith(('http://', 'https://', 'file://')):
                    report.add_warning(
                        code=EnvironmentSnapshotWarningCode.SUSPICIOUS_PACKAGE_URL,
                        detail=f"Package '{pkg_name}' has suspicious URL: {url}",
                        severity=WarningSeverity.LOW,
                        context={"package": pkg_name, "url": url}
                    )
            
            # Validate hash if present
            if 'hash' in pkg:
                if isinstance(pkg['hash'], dict):
                    if 'md5' in pkg['hash']:
                        md5 = pkg['hash']['md5']
                        if not re.match(r'^[a-fA-F0-9]{32}$', md5):
                            report.add_warning(
                                code=EnvironmentSnapshotWarningCode.INVALID_MD5_HASH,
                                detail=f"Package '{pkg_name}' has invalid MD5 hash format",
                                severity=WarningSeverity.MEDIUM,
                                context={"package": pkg_name, "md5": md5}
                            )
                    if 'sha256' in pkg['hash']:
                        sha256 = pkg['hash']['sha256']
                        if not re.match(r'^[a-fA-F0-9]{64}$', sha256):
                            report.add_warning(
                                code=EnvironmentSnapshotWarningCode.INVALID_SHA256_HASH,
                                detail=f"Package '{pkg_name}' has invalid SHA256 hash format",
                                severity=WarningSeverity.MEDIUM,
                                context={"package": pkg_name, "sha256": sha256}
                            )
    
    
    # ========================================================================
    # NON-BLOCKING Validation Rule Functions (Warnings)
    # ========================================================================
    
    def _check_system_packages_pinned(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Check that system packages are pinned to exact versions."""
        for pkg in target.system_packages:
            # Check for common version patterns
            has_version = (
                '=' in pkg or 
                re.match(r'^.+-[\d.]+', pkg) or
                re.match(r'^.+:[\d.]+', pkg)
            )
            
            if not has_version:
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.UNPINNED_SYSTEM_PACKAGE,
                    detail=f"System package may not be pinned to exact version: '{pkg}'. "
                           f"Expected format: 'package=1.2.3' or 'package-1.2.3'",
                    severity=WarningSeverity.MEDIUM,
                    context={"package": pkg}
                )
            
            # Check for version ranges
            if any(char in pkg for char in ['>', '<', '~', '*']):
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.VERSION_RANGE_SYSTEM_PACKAGE,
                    detail=f"System package should not use version ranges: '{pkg}'",
                    severity=WarningSeverity.HIGH,
                    context={"package": pkg}
                )
    
    def _check_python_version_stability(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Warn about unstable Python versions."""
        try:
            version = Version(target.python_version)
            
            if version.is_prerelease:
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.PRERELEASE_PYTHON_VERSION,
                    detail=f"Python version {target.python_version} is a pre-release and "
                           f"may not be stable",
                    severity=WarningSeverity.HIGH,
                    context={"python_version": target.python_version}
                )
            
            # Check for very old or very new versions
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.DEPRECATED_PYTHON_VERSION,
                    detail=f"Python version {target.python_version} is deprecated or "
                           f"end-of-life",
                    severity=WarningSeverity.HIGH,
                    context={"python_version": target.python_version}
                )
                
        except InvalidVersion:
            # This should have been caught by validate_python_version_format
            pass
    
    def _check_dependency_hashes(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Warn if dependencies don't include integrity hashes."""
        if target.snapshot_type == EnvSnapshotType.PIP_LOCK:
            lines = target.specification.strip().split('\n')
            has_hash = False
            dependency_count = 0
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    dependency_count += 1
                    if '--hash' in line or 'hash=' in line:
                        has_hash = True
            
            if dependency_count > 0 and not has_hash:
                report.add_warning(
                    code=EnvironmentSnapshotWarningCode.MISSING_PACKAGE_HASHES,
                    detail=f"No package hashes found in PIP lock file. "
                           f"Hashes are recommended for security and reproducibility",
                    severity=WarningSeverity.MEDIUM,
                    context={"dependency_count": dependency_count}
                )
        
        elif target.snapshot_type == EnvSnapshotType.CONDA_LOCK:
            try:
                data = json.loads(target.specification)
                packages = data.get('package', [])
                
                packages_with_hash = sum(
                    1 for pkg in packages 
                    if isinstance(pkg, dict) and 'hash' in pkg
                )
                
                if packages_with_hash < len(packages):
                    report.add_warning(
                        code=EnvironmentSnapshotWarningCode.MISSING_CONDA_HASHES,
                        detail=f"{len(packages) - packages_with_hash} out of {len(packages)} "
                               f"packages missing integrity hashes",
                        severity=WarningSeverity.LOW,
                        context={
                            "total_packages": len(packages),
                            "packages_with_hash": packages_with_hash
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                # Already reported in format validation
                pass
    
    def _check_redundant_dependencies(
        self, 
        target: 'EnvironmentSnapshot', 
        report: ValidationReport, 
        rule_name: str
    ) -> None:
        """Check for potentially redundant or conflicting dependencies."""
        if target.snapshot_type == EnvSnapshotType.PIP_LOCK:
            lines = target.specification.strip().split('\n')
            package_names = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    try:
                        req = Requirement(line)
                        package_names.append(req.name.lower())
                    except Exception:
                        continue
            
            # Check for duplicates
            from collections import Counter
            name_counts = Counter(package_names)
            
            for name, count in name_counts.items():
                if count > 1:
                    report.add_warning(
                        code=EnvironmentSnapshotWarningCode.DUPLICATE_DEPENDENCY,
                        detail=f"Package '{name}' appears {count} times in requirements",
                        severity=WarningSeverity.HIGH,
                        context={"package": name, "count": count}
                    )
        
        elif target.snapshot_type == EnvSnapshotType.CONDA_LOCK:
            try:
                data = json.loads(target.specification)
                packages = data.get('package', [])
                
                package_names = [
                    pkg['name'].lower() 
                    for pkg in packages 
                    if isinstance(pkg, dict) and 'name' in pkg
                ]
                
                from collections import Counter
                name_counts = Counter(package_names)
                
                for name, count in name_counts.items():
                    if count > 1:
                        report.add_warning(
                            code=EnvironmentSnapshotWarningCode.DUPLICATE_DEPENDENCY,
                            detail=f"Package '{name}' appears {count} times in conda-lock",
                            severity=WarningSeverity.HIGH,
                            context={"package": name, "count": count}
                        )
            except (json.JSONDecodeError, KeyError):
                pass

