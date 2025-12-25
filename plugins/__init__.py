"""Plugin architecture for extensible functionality."""

from plugins.plugin_manager import (
    PluginManager,
    Plugin,
    PluginMetadata,
    PluginHook,
    plugin,
)

__all__ = [
    "PluginManager",
    "Plugin",
    "PluginMetadata",
    "PluginHook",
    "plugin",
]
