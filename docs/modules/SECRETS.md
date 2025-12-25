# Secrets Management Module

HashiCorp Vault integration for secure secrets management.

## Features

- **Multiple Auth Methods**: Token, Kubernetes, AppRole
- **KV v2 Support**: Versioned secrets
- **Async Client**: Non-blocking operations
- **Namespace Support**: Multi-tenant isolation

## Components

### VaultConfig
```python
from secrets_ import VaultConfig

config = VaultConfig(
    url="http://vault:8200",
    namespace="robotics-lab",
    mount_point="secret",
    auth_method="kubernetes",
)
```

### VaultClient
```python
from secrets_ import VaultClient, VaultConfig

config = VaultConfig(token="hvs.xxx")

async with VaultClient(config) as client:
    # Get secret
    secret = await client.get_secret("database/credentials")
    password = secret["password"]
    
    # Set secret
    await client.set_secret("api/keys", {
        "nvidia_api_key": "xxx",
        "aws_access_key": "yyy",
    })
```

## Authentication Methods

### Token Auth
```python
config = VaultConfig(
    auth_method="token",
    token="hvs.xxxxx",
)
```

### Kubernetes Auth
```python
config = VaultConfig(
    auth_method="kubernetes",
    # Uses service account JWT automatically
)
```

### AppRole Auth
```python
config = VaultConfig(
    auth_method="approle",
    role_id="xxx",
    secret_id="yyy",
)
```

## Convenience Functions
```python
from secrets_ import get_secret, set_secret

# Quick access without managing client
secret = await get_secret("nvidia/api-key")
await set_secret("robot/credentials", {"token": "xxx"})
```

## Security Best Practices

1. Use Kubernetes auth in clusters
2. Rotate tokens regularly
3. Use namespaces for isolation
4. Enable audit logging
5. Apply least-privilege policies
