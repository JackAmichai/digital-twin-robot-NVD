"""Compliance reporting and audit trail management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
import json
from pathlib import Path


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO_10218 = "iso_10218"  # Robot safety
    IEC_62443 = "iec_62443"  # Industrial cybersecurity


class CheckStatus(Enum):
    """Compliance check status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class AuditAction(Enum):
    """Audit log action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    COMMAND = "command"
    CONFIG_CHANGE = "config_change"


@dataclass
class AuditLog:
    """Audit log entry."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action: AuditAction = AuditAction.READ
    user_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    success: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "success": self.success,
        }


@dataclass
class ComplianceCheck:
    """Individual compliance check."""
    
    id: str
    name: str
    description: str
    standard: ComplianceStandard
    status: CheckStatus = CheckStatus.NOT_APPLICABLE
    evidence: str = ""
    remediation: str = ""


@dataclass
class ComplianceReport:
    """Compliance report with checks."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    standard: ComplianceStandard = ComplianceStandard.ISO_27001
    generated_at: datetime = field(default_factory=datetime.utcnow)
    checks: list[ComplianceCheck] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        applicable = [c for c in self.checks if c.status != CheckStatus.NOT_APPLICABLE]
        if not applicable:
            return 0.0
        passed = sum(1 for c in applicable if c.status == CheckStatus.PASS)
        return passed / len(applicable)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "standard": self.standard.value,
            "generated_at": self.generated_at.isoformat(),
            "pass_rate": self.pass_rate,
            "summary": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.status == CheckStatus.PASS),
                "failed": sum(1 for c in self.checks if c.status == CheckStatus.FAIL),
                "warnings": sum(1 for c in self.checks if c.status == CheckStatus.WARNING),
            },
            "checks": [
                {
                    "id": c.id,
                    "name": c.name,
                    "status": c.status.value,
                    "evidence": c.evidence,
                }
                for c in self.checks
            ],
        }


class ComplianceReporter:
    """Generate compliance reports and manage audit logs."""
    
    def __init__(self, storage_path: Path | str = "./compliance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._audit_logs: list[AuditLog] = []
        self._reports: dict[str, ComplianceReport] = {}
    
    def log_audit(self, log: AuditLog) -> None:
        """Record audit log entry."""
        self._audit_logs.append(log)
        self._persist_audit_log(log)
    
    def _persist_audit_log(self, log: AuditLog) -> None:
        """Persist audit log to file."""
        date_str = log.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_path / f"audit-{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")
    
    def get_audit_logs(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
    ) -> list[AuditLog]:
        """Query audit logs with filters."""
        logs = self._audit_logs
        
        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]
        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if action:
            logs = [l for l in logs if l.action == action]
        
        return logs
    
    def generate_report(
        self,
        standard: ComplianceStandard,
    ) -> ComplianceReport:
        """Generate compliance report."""
        checks = self._get_checks_for_standard(standard)
        report = ComplianceReport(standard=standard, checks=checks)
        
        self._reports[report.id] = report
        self._save_report(report)
        
        return report
    
    def _get_checks_for_standard(
        self,
        standard: ComplianceStandard,
    ) -> list[ComplianceCheck]:
        """Get compliance checks for standard."""
        checks_map = {
            ComplianceStandard.ISO_27001: [
                ComplianceCheck("A.5.1", "Information Security Policies", "Security policy defined", standard),
                ComplianceCheck("A.6.1", "Internal Organization", "Roles defined", standard),
                ComplianceCheck("A.9.1", "Access Control Policy", "Access control implemented", standard),
                ComplianceCheck("A.12.1", "Operational Procedures", "Procedures documented", standard),
            ],
            ComplianceStandard.SOC2: [
                ComplianceCheck("CC1", "Control Environment", "Control environment established", standard),
                ComplianceCheck("CC2", "Communication", "Communication policies", standard),
                ComplianceCheck("CC6", "Logical Access", "Access controls", standard),
                ComplianceCheck("CC7", "System Operations", "Operational monitoring", standard),
            ],
            ComplianceStandard.ISO_10218: [
                ComplianceCheck("4.1", "Robot Design", "Safety design requirements", standard),
                ComplianceCheck("4.2", "Safety Functions", "Emergency stop implemented", standard),
                ComplianceCheck("5.1", "Hazard Identification", "Risk assessment complete", standard),
                ComplianceCheck("5.2", "Safety Distance", "Safe operating envelope defined", standard),
            ],
        }
        return checks_map.get(standard, [])
    
    def _save_report(self, report: ComplianceReport) -> None:
        """Save report to file."""
        report_file = self.storage_path / f"report-{report.id}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def get_report(self, report_id: str) -> ComplianceReport | None:
        """Get report by ID."""
        return self._reports.get(report_id)
    
    def list_reports(
        self,
        standard: ComplianceStandard | None = None,
    ) -> list[ComplianceReport]:
        """List generated reports."""
        reports = list(self._reports.values())
        if standard:
            reports = [r for r in reports if r.standard == standard]
        return reports
