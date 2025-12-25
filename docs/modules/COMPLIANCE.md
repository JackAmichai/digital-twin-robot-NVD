# Compliance Reporting Module

Regulatory compliance and audit trail management.

## Features

- **Multiple Standards**: ISO 27001, SOC2, GDPR, HIPAA, ISO 10218
- **Audit Logging**: Immutable action trail
- **Report Generation**: Automated compliance reports
- **Evidence Collection**: Attach evidence to checks

## Compliance Standards

| Standard | Description |
|----------|-------------|
| ISO 27001 | Information security management |
| SOC2 | Service organization controls |
| GDPR | EU data protection |
| HIPAA | Healthcare data protection |
| ISO 10218 | Robot safety requirements |
| IEC 62443 | Industrial cybersecurity |

## Usage

### Audit Logging
```python
from compliance import ComplianceReporter, AuditLog, AuditAction

reporter = ComplianceReporter("./compliance_data")

# Log user action
reporter.log_audit(AuditLog(
    action=AuditAction.COMMAND,
    user_id="user-123",
    resource_type="robot",
    resource_id="R001",
    details={"command": "move_to", "params": {"x": 10, "y": 20}},
    ip_address="192.168.1.50",
))

# Log configuration change
reporter.log_audit(AuditLog(
    action=AuditAction.CONFIG_CHANGE,
    user_id="admin",
    resource_type="system",
    details={"setting": "max_speed", "old": 1.0, "new": 1.5},
))
```

### Query Audit Logs
```python
from datetime import datetime, timedelta

logs = reporter.get_audit_logs(
    start_date=datetime.utcnow() - timedelta(days=30),
    user_id="admin",
    action=AuditAction.CONFIG_CHANGE,
)

for log in logs:
    print(f"{log.timestamp}: {log.action.value} by {log.user_id}")
```

### Generate Reports
```python
from compliance import ComplianceStandard

report = reporter.generate_report(ComplianceStandard.ISO_10218)

print(f"Pass Rate: {report.pass_rate:.1%}")
print(f"Checks: {len(report.checks)}")

for check in report.checks:
    status = "✓" if check.status == CheckStatus.PASS else "✗"
    print(f"{status} {check.id}: {check.name}")
```

## Report Output
```json
{
    "id": "rpt-123",
    "standard": "iso_10218",
    "pass_rate": 0.75,
    "summary": {
        "total": 4,
        "passed": 3,
        "failed": 1,
        "warnings": 0
    }
}
```

## Integration

- SIEM integration (Splunk, ELK)
- GRC platforms
- Automated evidence collection
- Scheduled report generation
