"""
SAST Scanner - Static Application Security Testing.
Integrates Bandit for Python security analysis.
"""

import subprocess
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """Vulnerability severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Confidence(Enum):
    """Finding confidence levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SASTFinding:
    """Single SAST vulnerability finding."""
    rule_id: str
    severity: Severity
    confidence: Confidence
    file_path: str
    line_number: int
    code_snippet: str
    message: str
    cwe_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "message": self.message,
            "cwe_id": self.cwe_id,
        }


@dataclass
class SASTResult:
    """SAST scan results."""
    scan_id: str
    target_path: str
    findings: List[SASTFinding] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    scan_duration_ms: float = 0.0
    
    @property
    def high_severity_count(self) -> int:
        return sum(1 for f in self.findings 
                   if f.severity in (Severity.HIGH, Severity.CRITICAL))
    
    @property
    def passed(self) -> bool:
        """Check if scan passed (no high/critical findings)."""
        return self.high_severity_count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "target_path": self.target_path,
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
            "scan_duration_ms": self.scan_duration_ms,
            "summary": {
                "total_findings": len(self.findings),
                "high_severity": self.high_severity_count,
                "passed": self.passed,
            }
        }


class SASTScanner:
    """
    Static Application Security Testing scanner.
    Uses Bandit for Python code analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.exclude_dirs = self.config.get("exclude_dirs", [
            ".git", "__pycache__", "venv", "node_modules", ".tox"
        ])
        self.severity_threshold = self.config.get(
            "severity_threshold", Severity.MEDIUM
        )
    
    def scan(self, target_path: str) -> SASTResult:
        """
        Run SAST scan on target path.
        
        Args:
            target_path: Directory or file to scan
            
        Returns:
            SASTResult with findings
        """
        import time
        import uuid
        
        start_time = time.time()
        scan_id = str(uuid.uuid4())[:8]
        result = SASTResult(scan_id=scan_id, target_path=target_path)
        
        try:
            findings = self._run_bandit(target_path)
            result.findings = findings
        except Exception as e:
            result.errors.append(str(e))
        
        result.scan_duration_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_bandit(self, target_path: str) -> List[SASTFinding]:
        """Run Bandit scanner and parse results."""
        exclude_arg = ",".join(self.exclude_dirs)
        
        cmd = [
            "bandit",
            "-r", target_path,
            "-f", "json",
            "--exclude", exclude_arg,
            "-ll",  # Only medium and higher
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if proc.stdout:
                return self._parse_bandit_output(proc.stdout)
            return []
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Bandit scan timed out")
        except FileNotFoundError:
            raise RuntimeError("Bandit not installed. Run: pip install bandit")
    
    def _parse_bandit_output(self, output: str) -> List[SASTFinding]:
        """Parse Bandit JSON output into findings."""
        findings = []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return findings
        
        for result in data.get("results", []):
            severity = Severity[result.get("issue_severity", "LOW").upper()]
            confidence = Confidence[result.get("issue_confidence", "LOW").upper()]
            
            finding = SASTFinding(
                rule_id=result.get("test_id", "unknown"),
                severity=severity,
                confidence=confidence,
                file_path=result.get("filename", ""),
                line_number=result.get("line_number", 0),
                code_snippet=result.get("code", ""),
                message=result.get("issue_text", ""),
                cwe_id=result.get("issue_cwe", {}).get("id"),
            )
            findings.append(finding)
        
        return findings
    
    def scan_with_semgrep(self, target_path: str) -> SASTResult:
        """
        Alternative scan using Semgrep for broader coverage.
        """
        import time
        import uuid
        
        start_time = time.time()
        scan_id = str(uuid.uuid4())[:8]
        result = SASTResult(scan_id=scan_id, target_path=target_path)
        
        cmd = [
            "semgrep",
            "--config", "auto",
            "--json",
            target_path,
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if proc.stdout:
                result.findings = self._parse_semgrep_output(proc.stdout)
                
        except subprocess.TimeoutExpired:
            result.errors.append("Semgrep scan timed out")
        except FileNotFoundError:
            result.errors.append("Semgrep not installed")
        
        result.scan_duration_ms = (time.time() - start_time) * 1000
        return result
    
    def _parse_semgrep_output(self, output: str) -> List[SASTFinding]:
        """Parse Semgrep JSON output."""
        findings = []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return findings
        
        severity_map = {
            "ERROR": Severity.HIGH,
            "WARNING": Severity.MEDIUM,
            "INFO": Severity.LOW,
        }
        
        for result in data.get("results", []):
            severity = severity_map.get(
                result.get("extra", {}).get("severity", "INFO"),
                Severity.LOW
            )
            
            finding = SASTFinding(
                rule_id=result.get("check_id", "unknown"),
                severity=severity,
                confidence=Confidence.HIGH,
                file_path=result.get("path", ""),
                line_number=result.get("start", {}).get("line", 0),
                code_snippet=result.get("extra", {}).get("lines", ""),
                message=result.get("extra", {}).get("message", ""),
            )
            findings.append(finding)
        
        return findings
