"""Plugin management system with hooks and lifecycle."""

import importlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar
from functools import wraps


class PluginHook(Enum):
    """Available plugin hooks."""
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    PRE_TASK = "pre_task"
    POST_TASK = "post_task"
    PRE_COMMAND = "pre_command"
    POST_COMMAND = "post_command"
    ON_ERROR = "on_error"
    ON_ROBOT_CONNECT = "on_robot_connect"
    ON_ROBOT_DISCONNECT = "on_robot_disconnect"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    hooks: list[PluginHook] = field(default_factory=list)


class Plugin(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass
    
    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass
    
    async def on_startup(self) -> None:
        """Called on application startup."""
        pass
    
    async def on_shutdown(self) -> None:
        """Called on application shutdown."""
        pass
    
    async def pre_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Called before task execution."""
        return task
    
    async def post_task(self, task: dict[str, Any], result: Any) -> Any:
        """Called after task execution."""
        return result
    
    async def on_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Called when an error occurs."""
        pass


T = TypeVar("T", bound=Plugin)

# Plugin registry
_registered_plugins: dict[str, type[Plugin]] = {}


def plugin(cls: type[T]) -> type[T]:
    """Decorator to register a plugin class."""
    _registered_plugins[cls.__name__] = cls
    return cls


class PluginManager:
    """Manage plugin lifecycle and execution."""
    
    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._hooks: dict[PluginHook, list[Plugin]] = {h: [] for h in PluginHook}
    
    def load_plugin(self, plugin_class: type[Plugin]) -> Plugin:
        """Load and initialize a plugin."""
        instance = plugin_class()
        name = instance.metadata.name
        
        # Check dependencies
        for dep in instance.metadata.dependencies:
            if dep not in self._plugins:
                raise RuntimeError(f"Missing dependency: {dep}")
        
        instance.on_load()
        self._plugins[name] = instance
        
        # Register hooks
        for hook in instance.metadata.hooks:
            self._hooks[hook].append(instance)
        
        return instance
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        
        plugin.on_unload()
        
        # Remove from hooks
        for hook_list in self._hooks.values():
            if plugin in hook_list:
                hook_list.remove(plugin)
        
        del self._plugins[name]
        return True
    
    def get_plugin(self, name: str) -> Plugin | None:
        """Get plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> list[PluginMetadata]:
        """List all loaded plugins."""
        return [p.metadata for p in self._plugins.values()]
    
    async def execute_hook(
        self,
        hook: PluginHook,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute all plugins registered for a hook."""
        results = []
        for plugin in self._hooks[hook]:
            handler = getattr(plugin, hook.value, None)
            if handler:
                result = await handler(*args, **kwargs)
                results.append(result)
        return results
    
    def load_from_directory(self, directory: Path) -> list[Plugin]:
        """Load all plugins from a directory."""
        loaded = []
        
        for path in directory.glob("*.py"):
            if path.name.startswith("_"):
                continue
            
            spec = importlib.util.spec_from_file_location(
                path.stem,
                path,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find Plugin subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, Plugin)
                        and attr is not Plugin
                    ):
                        plugin = self.load_plugin(attr)
                        loaded.append(plugin)
        
        return loaded
    
    async def startup(self) -> None:
        """Run startup hooks."""
        await self.execute_hook(PluginHook.STARTUP)
    
    async def shutdown(self) -> None:
        """Run shutdown hooks."""
        await self.execute_hook(PluginHook.SHUTDOWN)
