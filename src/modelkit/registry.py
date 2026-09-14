"""Dynamic Discovery & Registry: modelkit/registry.py

Binds string identifiers declared in mlkit.json to user subclasses dynamically.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

# Registry mapping: category -> {name: registered_class}
_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {}


def register(category: str, name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator that registers a class under a functional domain.

    Args:
        category: Functional domain (e.g. "model", "dataset", "trainer", "evaluator", "inference").
        name: Unique identifier string for the class.

    Returns:
        Decorator callable returning the original class reference.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        _REGISTRY[category][name] = cls
        return cls

    return decorator


def get(category: str, name: str) -> Type[Any]:
    """Looks up and returns the registered class reference by name.

    Args:
        category: Functional domain category.
        name: Registered identifier name.

    Returns:
        The registered class reference.

    Raises:
        KeyError: If the requested category or identifier has not been registered.
    """
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list(_REGISTRY.get(category, {}).keys())
        raise KeyError(
            f"No registered item found for category '{category}' with name '{name}'. "
            f"Available items in '{category}': {available}"
        )
    return _REGISTRY[category][name]


def clear_registry(category: Optional[str] = None) -> None:
    """Utility helper to clear registrations (primarily for testing)."""
    if category is not None:
        _REGISTRY.pop(category, None)
    else:
        _REGISTRY.clear()
