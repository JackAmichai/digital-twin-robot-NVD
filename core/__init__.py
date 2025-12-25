# Core Module - Dependency Injection
"""
IoC container for better testability.
"""

from .container import Container, inject, singleton

__all__ = [
    "Container",
    "inject",
    "singleton",
]
