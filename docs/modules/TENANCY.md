# Multi-Tenancy Module

Tenant isolation and resource management for SaaS deployments.

## Features

- **Tenant Isolation**: Context-based data separation
- **Tier Management**: Free, Standard, Enterprise
- **Quota Enforcement**: Resource limits per tenant
- **Context Variables**: Thread-safe tenant context

## Components

### Tenant
```python
from tenancy import Tenant, TenantTier

# Create with tier-based defaults
tenant = Tenant.from_tier("Acme Corp", TenantTier.ENTERPRISE)

print(tenant.quota.max_robots)  # 100
print(tenant.quota.gpu_hours_per_month)  # 500.0
```

### TenantContext
```python
from tenancy import TenantContext, current_tenant

with TenantContext(tenant):
    # All operations scoped to tenant
    current = current_tenant()
    print(current.name)  # "Acme Corp"
```

### TenantManager
```python
from tenancy import TenantManager, TenantTier

manager = TenantManager()

# Create tenant
tenant = manager.create_tenant("Acme Corp", TenantTier.STANDARD)

# Check quotas
can_add = manager.check_quota(tenant.id, "robots", 5)

# Upgrade tier
manager.update_tier(tenant.id, TenantTier.ENTERPRISE)
```

## Subscription Tiers

| Feature | Free | Standard | Enterprise |
|---------|------|----------|------------|
| Robots | 2 | 10 | 100 |
| API Calls/Hour | 100 | 5,000 | 100,000 |
| Storage (GB) | 1 | 50 | 1,000 |
| GPU Hours/Month | 0 | 10 | 500 |

## Decorator Usage
```python
from tenancy import require_tenant

@require_tenant
def get_robots():
    tenant = current_tenant()
    return db.query(Robot).filter(tenant_id=tenant.id).all()
```

## Integration

- Database schema isolation
- API authentication middleware
- Kubernetes namespace mapping
- Resource quota enforcement
