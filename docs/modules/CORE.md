# Dependency Injection Module

## Overview
IoC container for better testability and loose coupling.

## Files

### `container.py`
Simple dependency injection container.

```python
class Container:
    def register(interface, implementation)
    def register_instance(interface, instance)
    def register_factory(interface, factory)
    def register_singleton(interface, implementation)
    def resolve(interface) -> T
    def clear()
```

**Decorators:**
- `@inject`: Auto-inject function parameters
- `@singleton`: Register class as singleton

## Usage

```python
from core import Container, inject, singleton

# Define interfaces and implementations
class IDatabase:
    def query(self, sql): ...

class PostgresDB(IDatabase):
    def query(self, sql):
        return f"Postgres: {sql}"

class MockDB(IDatabase):
    def query(self, sql):
        return f"Mock: {sql}"

# Register dependencies
container = Container.get_instance()
container.register(IDatabase, PostgresDB)

# For testing, swap implementation
container.register(IDatabase, MockDB)

# Resolve dependencies
db = container.resolve(IDatabase)

# Auto-injection with decorator
@inject
def process_data(db: IDatabase):
    return db.query("SELECT * FROM data")

# Singleton pattern
@singleton
class ConfigService:
    def __init__(self):
        self.config = load_config()
```

## Benefits
- Loose coupling between components
- Easy testing with mock implementations
- Centralized dependency management
- Auto-wiring via type annotations
