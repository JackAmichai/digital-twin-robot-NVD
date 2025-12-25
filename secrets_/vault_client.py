"""HashiCorp Vault client for secrets management."""

import os
from dataclasses import dataclass
from typing import Any
import aiohttp


class SecretNotFoundError(Exception):
    """Raised when a secret is not found."""
    pass


@dataclass
class VaultConfig:
    """Vault connection configuration."""
    
    url: str = "http://localhost:8200"
    token: str | None = None
    namespace: str = "robotics-lab"
    mount_point: str = "secret"
    auth_method: str = "token"  # token, kubernetes, approle
    role_id: str | None = None
    secret_id: str | None = None
    
    def __post_init__(self) -> None:
        """Load token from environment if not provided."""
        if not self.token:
            self.token = os.environ.get("VAULT_TOKEN")


class VaultClient:
    """Async HashiCorp Vault client."""
    
    def __init__(self, config: VaultConfig | None = None):
        self.config = config or VaultConfig()
        self._token: str | None = self.config.token
        self._session: aiohttp.ClientSession | None = None
    
    async def __aenter__(self) -> "VaultClient":
        """Enter async context."""
        await self.connect()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()
    
    async def connect(self) -> None:
        """Initialize connection and authenticate."""
        self._session = aiohttp.ClientSession(
            headers=self._get_headers(),
        )
        
        if self.config.auth_method == "kubernetes":
            await self._authenticate_kubernetes()
        elif self.config.auth_method == "approle":
            await self._authenticate_approle()
    
    async def close(self) -> None:
        """Close session."""
        if self._session:
            await self._session.close()
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["X-Vault-Token"] = self._token
        if self.config.namespace:
            headers["X-Vault-Namespace"] = self.config.namespace
        return headers
    
    async def _authenticate_kubernetes(self) -> None:
        """Authenticate using Kubernetes service account."""
        jwt_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        with open(jwt_path) as f:
            jwt = f.read()
        
        async with self._session.post(
            f"{self.config.url}/v1/auth/kubernetes/login",
            json={"jwt": jwt, "role": "robotics-lab"},
        ) as resp:
            data = await resp.json()
            self._token = data["auth"]["client_token"]
    
    async def _authenticate_approle(self) -> None:
        """Authenticate using AppRole."""
        async with self._session.post(
            f"{self.config.url}/v1/auth/approle/login",
            json={
                "role_id": self.config.role_id,
                "secret_id": self.config.secret_id,
            },
        ) as resp:
            data = await resp.json()
            self._token = data["auth"]["client_token"]
    
    async def get_secret(self, path: str) -> dict[str, Any]:
        """Retrieve secret from Vault."""
        url = f"{self.config.url}/v1/{self.config.mount_point}/data/{path}"
        
        async with self._session.get(url, headers=self._get_headers()) as resp:
            if resp.status == 404:
                raise SecretNotFoundError(f"Secret not found: {path}")
            resp.raise_for_status()
            data = await resp.json()
            return data["data"]["data"]
    
    async def set_secret(
        self,
        path: str,
        data: dict[str, Any],
    ) -> None:
        """Store secret in Vault."""
        url = f"{self.config.url}/v1/{self.config.mount_point}/data/{path}"
        
        async with self._session.post(
            url,
            json={"data": data},
            headers=self._get_headers(),
        ) as resp:
            resp.raise_for_status()
    
    async def delete_secret(self, path: str) -> None:
        """Delete secret from Vault."""
        url = f"{self.config.url}/v1/{self.config.mount_point}/data/{path}"
        
        async with self._session.delete(
            url,
            headers=self._get_headers(),
        ) as resp:
            resp.raise_for_status()


# Module-level convenience functions
_client: VaultClient | None = None


async def get_secret(path: str, config: VaultConfig | None = None) -> dict[str, Any]:
    """Get secret using shared client."""
    global _client
    if _client is None:
        _client = VaultClient(config)
        await _client.connect()
    return await _client.get_secret(path)


async def set_secret(
    path: str,
    data: dict[str, Any],
    config: VaultConfig | None = None,
) -> None:
    """Set secret using shared client."""
    global _client
    if _client is None:
        _client = VaultClient(config)
        await _client.connect()
    await _client.set_secret(path, data)
