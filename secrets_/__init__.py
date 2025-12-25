"""Secrets management with HashiCorp Vault integration."""

from secrets_.vault_client import (
    VaultClient,
    VaultConfig,
    SecretNotFoundError,
    get_secret,
    set_secret,
)

__all__ = [
    "VaultClient",
    "VaultConfig",
    "SecretNotFoundError",
    "get_secret",
    "set_secret",
]
