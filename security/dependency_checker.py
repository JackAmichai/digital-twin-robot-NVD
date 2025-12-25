"""
Dependency Checker - Scan dependencies for known vulnerabilities.
Uses safety and pip-audit for Python dependency scanning.
"""

import subprocess
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class VulnSeverity(Enum):
    """Vulnerability severity."""
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VulnerableDependency:
    """A vulnerable dependency."""
    package_name: str
    installed_version: str
    vulnerability_id: str
    severity: VulnSeverity
    description: str
    fixed_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_name": self.package_name,
            "installed_version": self.installed_version,
            "vulnerability_id": self.vulnerability_id,
            "severity": self.severity.value,
            "description": self.description,
            "fixed_version": self.fixed_version,
        }


@dataclass
class DependencyCheckResult:
    """Dependency check results."""
    scan_id: str
    vulnerabilities: List[VulnerableDependency] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities 
                   if v.severity in (VulnSeverity.CRITICAL, VulnSeverity.HIGH))
    
    @property
    def passed(self) -> bool:
        return self.critical_count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "errors": self.errors,
            "summary": {
                "total": len(self.vulnerabilities),
                "critical_high": self.critical_count,
                "passed": self.passed,
            }
        }


class DependencyChecker:
    """
    Check Python dependencies for known vulnerabilities.
    """
    
    def __init__(self):
        pass
    
    def check_with_safety(
        self,
        requirements_file: Optional[str] = None
    ) -> DependencyCheckResult:
        """
        Check dependencies using Safety.
        
        Args:
            requirements_file: Path to requirements.txt (optional)
        """
        import uuid
        
        scan_id = str(uuid.uuid4())[:8]
        result = DependencyCheckResult(scan_id=scan_id)
        
        cmd = ["safety", "check", "--json"]
        if requirements_file:
            cmd.extend(["-r", requirements_file])
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            # Safety returns exit code 255 when vulnerabilities found
            if proc.stdout:
                result.vulnerabilities = self._parse_safety_output(proc.stdout)
                
        except subprocess.TimeoutExpired:
            result.errors.append("Safety scan timed out")
        except FileNotFoundError:
            result.errors.append("Safety not installed. Run: pip install safety")
        
        return result
    
    def _parse_safety_output(self, output: str) -> List[VulnerableDependency]:
        """Parse Safety JSON output."""
        vulns = []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return vulns
        
        for vuln in data.get("vulnerabilities", []):
            v = VulnerableDependency(
                package_name=vuln.get("package_name", ""),
                installed_version=vuln.get("analyzed_version", ""),
                vulnerability_id=vuln.get("vulnerability_id", ""),
                severity=VulnSeverity.MEDIUM,  # Safety doesn't provide severity
                description=vuln.get("advisory", ""),
                fixed_version=vuln.get("fixed_versions", [""])[0] if vuln.get("fixed_versions") else None,
            )
            vulns.append(v)
        
        return vulns
    
    def check_with_pip_audit(self) -> DependencyCheckResult:
        """
        Check dependencies using pip-audit.
        """
        import uuid
        
        scan_id = str(uuid.uuid4())[:8]
        result = DependencyCheckResult(scan_id=scan_id)
        
        cmd = ["pip-audit", "--format", "json"]
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            
            if proc.stdout:
                result.vulnerabilities = self._parse_pip_audit_output(proc.stdout)
                
        except subprocess.TimeoutExpired:
            result.errors.append("pip-audit scan timed out")
        except FileNotFoundError:
            result.errors.append("pip-audit not installed. Run: pip install pip-audit")
        
        return result
    
    def _parse_pip_audit_output(
        self,
        output: str
    ) -> List[VulnerableDependency]:
        """Parse pip-audit JSON output."""
        vulns = []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return vulns
        
        for dep in data:
            name = dep.get("name", "")
            version = dep.get("version", "")
            
            for vuln in dep.get("vulns", []):
                severity_map = {
                    "CRITICAL": VulnSeverity.CRITICAL,
                    "HIGH": VulnSeverity.HIGH,
                    "MODERATE": VulnSeverity.MEDIUM,
                    "LOW": VulnSeverity.LOW,
                }
                
                v = VulnerableDependency(
                    package_name=name,
                    installed_version=version,
                    vulnerability_id=vuln.get("id", ""),
                    severity=severity_map.get(
                        vuln.get("severity", "").upper(),
                        VulnSeverity.UNKNOWN
                    ),
                    description=vuln.get("description", ""),
                    fixed_version=vuln.get("fix_versions", [""])[0] if vuln.get("fix_versions") else None,
                )
                vulns.append(v)
        
        return vulns
    
    def check_container_image(self, image: str) -> DependencyCheckResult:
        """
        Check container image for vulnerabilities using Trivy.
        
        Args:
            image: Docker image name (e.g., "python:3.11")
        """
        import uuid
        
        scan_id = str(uuid.uuid4())[:8]
        result = DependencyCheckResult(scan_id=scan_id)
        
        cmd = [
            "trivy", "image",
            "--format", "json",
            "--severity", "HIGH,CRITICAL",
            image,
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if proc.stdout:
                result.vulnerabilities = self._parse_trivy_output(proc.stdout)
                
        except subprocess.TimeoutExpired:
            result.errors.append("Trivy scan timed out")
        except FileNotFoundError:
            result.errors.append("Trivy not installed")
        
        return result
    
    def _parse_trivy_output(self, output: str) -> List[VulnerableDependency]:
        """Parse Trivy JSON output."""
        vulns = []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return vulns
        
        severity_map = {
            "CRITICAL": VulnSeverity.CRITICAL,
            "HIGH": VulnSeverity.HIGH,
            "MEDIUM": VulnSeverity.MEDIUM,
            "LOW": VulnSeverity.LOW,
        }
        
        for result_item in data.get("Results", []):
            for vuln in result_item.get("Vulnerabilities", []):
                v = VulnerableDependency(
                    package_name=vuln.get("PkgName", ""),
                    installed_version=vuln.get("InstalledVersion", ""),
                    vulnerability_id=vuln.get("VulnerabilityID", ""),
                    severity=severity_map.get(
                        vuln.get("Severity", "").upper(),
                        VulnSeverity.UNKNOWN
                    ),
                    description=vuln.get("Title", ""),
                    fixed_version=vuln.get("FixedVersion"),
                )
                vulns.append(v)
        
        return vulns
