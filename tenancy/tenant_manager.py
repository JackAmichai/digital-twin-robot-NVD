"""Multi-tenancy management and context handling."""

import contextvars
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar
from uuid import uuid4


# Context variable for current tenant
_current_tenant: contextvars.ContextVar["Tenant | None"] = contextvars.ContextVar(
    "current_tenant",
    default=None,
)


class TenantTier(Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuota:
    """Resource quotas per tenant."""
    
    max_robots: int = 5
    max_api_calls_per_hour: int = 1000
    max_storage_gb: float = 10.0
    max_concurrent_tasks: int = 10
    gpu_hours_per_month: float = 0.0


@dataclass
class Tenant:
    """Tenant entity with configuration."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    tier: TenantTier = TenantTier.FREE
    quota: TenantQuota = field(default_factory=TenantQuota)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    @classmethod
    def from_tier(cls, name: str, tier: TenantTier) -> "Tenant":
        """Create tenant with tier-based quotas."""
        quotas = {
            TenantTier.FREE: TenantQuota(
                max_robots=2,
                max_api_calls_per_hour=100,
                max_storage_gb=1.0,
            ),
            TenantTier.STANDARD: TenantQuota(
                max_robots=10,
                max_api_calls_per_hour=5000,
                max_storage_gb=50.0,
                gpu_hours_per_month=10.0,
            ),
            TenantTier.ENTERPRISE: TenantQuota(
                max_robots=100,
                max_api_calls_per_hour=100000,
                max_storage_gb=1000.0,
                max_concurrent_tasks=100,
                gpu_hours_per_month=500.0,
            ),
        }
        return cls(name=name, tier=tier, quota=quotas[tier])


class TenantContext:
    """Context manager for tenant scope."""
    
    def __init__(self, tenant: Tenant):
        self.tenant = tenant
        self._token: contextvars.Token | None = None
    
    def __enter__(self) -> Tenant:
        """Enter tenant context."""
        self._token = _current_tenant.set(self.tenant)
        return self.tenant
    
    def __exit__(self, *args: Any) -> None:
        """Exit tenant context."""
        if self._token:
            _current_tenant.reset(self._token)


def current_tenant() -> Tenant | None:
    """Get current tenant from context."""
    return _current_tenant.get()


T = TypeVar("T")


def require_tenant(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator requiring tenant context."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        tenant = current_tenant()
        if tenant is None:
            raise RuntimeError("No tenant in context")
        return func(*args, **kwargs)
    return wrapper


class TenantManager:
    """Manage tenant lifecycle and data isolation."""
    
    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
    
    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
    ) -> Tenant:
        """Create new tenant."""
        tenant = Tenant.from_tier(name, tier)
        self._tenants[tenant.id] = tenant
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)
    
    def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        """List all tenants."""
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.active]
        return tenants
    
    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.active = False
            return True
        return False
    
    def update_tier(self, tenant_id: str, tier: TenantTier) -> bool:
        """Update tenant tier and quotas."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            new_tenant = Tenant.from_tier(tenant.name, tier)
            tenant.tier = tier
            tenant.quota = new_tenant.quota
            return True
        return False
    
    def check_quota(
        self,
        tenant_id: str,
        resource: str,
        requested: float,
    ) -> bool:
        """Check if request is within quota."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        
        limits = {
            "robots": tenant.quota.max_robots,
            "api_calls": tenant.quota.max_api_calls_per_hour,
            "storage": tenant.quota.max_storage_gb,
            "tasks": tenant.quota.max_concurrent_tasks,
            "gpu_hours": tenant.quota.gpu_hours_per_month,
        }
        
        limit = limits.get(resource, 0)
        return requested <= limit
