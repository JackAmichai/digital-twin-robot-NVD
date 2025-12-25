"""
DAST Scanner - Dynamic Application Security Testing.
Integrates OWASP ZAP for runtime security analysis.
"""

import subprocess
import json
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskLevel(Enum):
    """DAST risk levels."""
    INFORMATIONAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class DASTFinding:
    """Single DAST vulnerability finding."""
    alert_name: str
    risk_level: RiskLevel
    url: str
    method: str
    description: str
    solution: str
    reference: str = ""
    cwe_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_name": self.alert_name,
            "risk_level": self.risk_level.name,
            "url": self.url,
            "method": self.method,
            "description": self.description,
            "solution": self.solution,
            "reference": self.reference,
            "cwe_id": self.cwe_id,
        }


@dataclass
class DASTResult:
    """DAST scan results."""
    scan_id: str
    target_url: str
    findings: List[DASTFinding] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    scan_duration_ms: float = 0.0
    
    @property
    def high_risk_count(self) -> int:
        return sum(1 for f in self.findings 
                   if f.risk_level == RiskLevel.HIGH)
    
    @property
    def passed(self) -> bool:
        return self.high_risk_count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "target_url": self.target_url,
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
            "scan_duration_ms": self.scan_duration_ms,
            "summary": {
                "total_findings": len(self.findings),
                "high_risk": self.high_risk_count,
                "passed": self.passed,
            }
        }


class DASTScanner:
    """
    Dynamic Application Security Testing scanner.
    Uses OWASP ZAP for runtime security testing.
    """
    
    def __init__(
        self,
        zap_api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ):
        self.zap_api_url = zap_api_url.rstrip("/")
        self.api_key = api_key or ""
    
    def quick_scan(self, target_url: str) -> DASTResult:
        """
        Run a quick spider + passive scan.
        
        Args:
            target_url: URL to scan
            
        Returns:
            DASTResult with findings
        """
        import time
        import uuid
        
        start_time = time.time()
        scan_id = str(uuid.uuid4())[:8]
        result = DASTResult(scan_id=scan_id, target_url=target_url)
        
        try:
            # Spider the target
            self._spider(target_url)
            
            # Wait for spider to complete
            self._wait_for_spider()
            
            # Get passive scan results
            result.findings = self._get_alerts(target_url)
            
        except requests.RequestException as e:
            result.errors.append(f"ZAP API error: {e}")
        except Exception as e:
            result.errors.append(str(e))
        
        result.scan_duration_ms = (time.time() - start_time) * 1000
        return result
    
    def active_scan(self, target_url: str) -> DASTResult:
        """
        Run full active scan (more thorough but slower).
        """
        import time
        import uuid
        
        start_time = time.time()
        scan_id = str(uuid.uuid4())[:8]
        result = DASTResult(scan_id=scan_id, target_url=target_url)
        
        try:
            # Spider first
            self._spider(target_url)
            self._wait_for_spider()
            
            # Start active scan
            scan_response = requests.get(
                f"{self.zap_api_url}/JSON/ascan/action/scan/",
                params={
                    "apikey": self.api_key,
                    "url": target_url,
                }
            )
            scan_response.raise_for_status()
            
            # Wait for active scan
            self._wait_for_active_scan()
            
            # Get all alerts
            result.findings = self._get_alerts(target_url)
            
        except Exception as e:
            result.errors.append(str(e))
        
        result.scan_duration_ms = (time.time() - start_time) * 1000
        return result
    
    def _spider(self, target_url: str) -> None:
        """Start ZAP spider on target."""
        response = requests.get(
            f"{self.zap_api_url}/JSON/spider/action/scan/",
            params={
                "apikey": self.api_key,
                "url": target_url,
            }
        )
        response.raise_for_status()
    
    def _wait_for_spider(self, timeout: int = 120) -> None:
        """Wait for spider to complete."""
        import time
        
        start = time.time()
        while time.time() - start < timeout:
            response = requests.get(
                f"{self.zap_api_url}/JSON/spider/view/status/",
                params={"apikey": self.api_key}
            )
            status = response.json().get("status", "0")
            if int(status) >= 100:
                return
            time.sleep(2)
        
        raise TimeoutError("Spider scan timed out")
    
    def _wait_for_active_scan(self, timeout: int = 600) -> None:
        """Wait for active scan to complete."""
        import time
        
        start = time.time()
        while time.time() - start < timeout:
            response = requests.get(
                f"{self.zap_api_url}/JSON/ascan/view/status/",
                params={"apikey": self.api_key}
            )
            status = response.json().get("status", "0")
            if int(status) >= 100:
                return
            time.sleep(5)
        
        raise TimeoutError("Active scan timed out")
    
    def _get_alerts(self, target_url: str) -> List[DASTFinding]:
        """Get all alerts for target."""
        response = requests.get(
            f"{self.zap_api_url}/JSON/core/view/alerts/",
            params={
                "apikey": self.api_key,
                "baseurl": target_url,
            }
        )
        response.raise_for_status()
        
        findings = []
        for alert in response.json().get("alerts", []):
            risk_map = {
                "Informational": RiskLevel.INFORMATIONAL,
                "Low": RiskLevel.LOW,
                "Medium": RiskLevel.MEDIUM,
                "High": RiskLevel.HIGH,
            }
            
            finding = DASTFinding(
                alert_name=alert.get("alert", ""),
                risk_level=risk_map.get(alert.get("risk", ""), RiskLevel.LOW),
                url=alert.get("url", ""),
                method=alert.get("method", "GET"),
                description=alert.get("description", ""),
                solution=alert.get("solution", ""),
                reference=alert.get("reference", ""),
                cwe_id=alert.get("cweid"),
            )
            findings.append(finding)
        
        return findings
    
    def generate_report(
        self,
        result: DASTResult,
        format: str = "html"
    ) -> str:
        """Generate scan report."""
        if format == "html":
            return self._generate_html_report(result)
        elif format == "json":
            return json.dumps(result.to_dict(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _generate_html_report(self, result: DASTResult) -> str:
        """Generate HTML report."""
        findings_html = ""
        for f in result.findings:
            color = {
                RiskLevel.HIGH: "red",
                RiskLevel.MEDIUM: "orange",
                RiskLevel.LOW: "yellow",
                RiskLevel.INFORMATIONAL: "blue",
            }.get(f.risk_level, "gray")
            
            findings_html += f"""
            <div class="finding" style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0;">
                <h3>{f.alert_name}</h3>
                <p><strong>Risk:</strong> {f.risk_level.name}</p>
                <p><strong>URL:</strong> {f.url}</p>
                <p>{f.description}</p>
                <p><strong>Solution:</strong> {f.solution}</p>
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>DAST Report - {result.scan_id}</title></head>
        <body>
            <h1>DAST Security Scan Report</h1>
            <p>Target: {result.target_url}</p>
            <p>Findings: {len(result.findings)}</p>
            <p>High Risk: {result.high_risk_count}</p>
            <hr/>
            {findings_html}
        </body>
        </html>
        """
