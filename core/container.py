"""
Dependency Injection Container.
"""

from typing import Type, TypeVar, Dict, Any, Callable, Optional
from functools import wraps
import inspect

T = TypeVar("T")


class Container:
    """
    Simple IoC container for dependency injection.
    """
    
    _instance: Optional["Container"] = None
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
    
    @classmethod
    def get_instance(cls) -> "Container":
        """Get singleton container instance."""
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance
    
    def register(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register implementation for interface."""
        self._services[interface] = implementation
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register existing instance."""
        self._singletons[interface] = instance
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register factory function."""
        self._factories[interface] = factory
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register as singleton (lazy initialization)."""
        self._services[interface] = implementation
        self._singletons[interface] = None  # Mark as singleton
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve dependency."""
        # Check for existing singleton instance
        if interface in self._singletons:
            if self._singletons[interface] is not None:
                return self._singletons[interface]
            # Create singleton instance
            impl = self._services.get(interface)
            if impl:
                instance = self._create_instance(impl)
                self._singletons[interface] = instance
                return instance
        
        # Check for factory
        if interface in self._factories:
            return self._factories[interface]()
        
        # Check for registered implementation
        if interface in self._services:
            return self._create_instance(self._services[interface])
        
        # Try to create directly
        return self._create_instance(interface)
    
    def _create_instance(self, cls: Type[T]) -> T:
        """Create instance with dependency injection."""
        sig = inspect.signature(cls.__init__)
        params = {}
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            
            if param.annotation != inspect.Parameter.empty:
                # Resolve typed parameter
                params[name] = self.resolve(param.annotation)
            elif param.default != inspect.Parameter.empty:
                # Use default value
                params[name] = param.default
        
        return cls(**params)
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


def inject(func: Callable) -> Callable:
    """
    Decorator to inject dependencies into function parameters.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        container = Container.get_instance()
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            if name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[name] = container.resolve(param.annotation)
                except Exception:
                    pass
        
        return func(*args, **kwargs)
    
    return wrapper


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator to register class as singleton.
    """
    container = Container.get_instance()
    container.register_singleton(cls, cls)
    return cls


# Convenience functions
def get_container() -> Container:
    """Get the global container instance."""
    return Container.get_instance()


def register(interface: Type[T], implementation: Type[T]) -> None:
    """Register in global container."""
    Container.get_instance().register(interface, implementation)


def resolve(interface: Type[T]) -> T:
    """Resolve from global container."""
    return Container.get_instance().resolve(interface)
