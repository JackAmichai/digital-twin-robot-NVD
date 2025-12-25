# Plugin Architecture Module

Extensible plugin system for custom functionality.

## Features

- **Hook System**: Intercept and modify behavior
- **Lifecycle Management**: Load/unload at runtime
- **Dependency Resolution**: Plugin dependencies
- **Directory Loading**: Auto-discover plugins

## Plugin Hooks

| Hook | Description |
|------|-------------|
| `startup` | Application start |
| `shutdown` | Application stop |
| `pre_task` | Before task execution |
| `post_task` | After task execution |
| `on_error` | Error handling |
| `on_robot_connect` | Robot connection |
| `on_robot_disconnect` | Robot disconnection |

## Creating a Plugin

```python
from plugins import Plugin, PluginMetadata, PluginHook, plugin

@plugin
class LoggingPlugin(Plugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="logging-plugin",
            version="1.0.0",
            description="Enhanced logging",
            hooks=[PluginHook.PRE_TASK, PluginHook.POST_TASK],
        )
    
    async def pre_task(self, task):
        print(f"Starting task: {task['id']}")
        return task
    
    async def post_task(self, task, result):
        print(f"Completed task: {task['id']}")
        return result
```

## Plugin Manager

```python
from plugins import PluginManager
from pathlib import Path

manager = PluginManager()

# Load single plugin
manager.load_plugin(LoggingPlugin)

# Load from directory
manager.load_from_directory(Path("./plugins_dir"))

# Execute hook
results = await manager.execute_hook(
    PluginHook.PRE_TASK,
    task={"id": "T123"},
)

# Lifecycle
await manager.startup()
await manager.shutdown()
```

## Dependencies

```python
@plugin
class AdvancedPlugin(Plugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="advanced-plugin",
            version="1.0.0",
            dependencies=["logging-plugin"],  # Requires logging-plugin
            hooks=[PluginHook.PRE_TASK],
        )
```

## Best Practices

1. Keep plugins focused and single-purpose
2. Document hook behavior clearly
3. Handle errors gracefully in hooks
4. Use metadata for versioning
5. Test plugins in isolation
