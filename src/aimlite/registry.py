"""Dynamic Discovery & Registry: aimlite/registry.py

Binds string identifiers declared in aimlite.json to user subclasses dynamically.
Automatically tracks subclasses of Model, Dataset, BaseTrainer, BaseEvaluator,
BaseInference, and BaseConfig under the hood without requiring explicit decorators.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

# Registry mapping: category -> {name: registered_class}
_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {}


def register_class(category: str, cls: Type[Any], name: Optional[str] = None) -> None:
    """Registers a class under a functional domain without requiring a decorator.

    Args:
        category: Functional domain (e.g. "model", "dataset", "trainer", "evaluator", "inference", "config").
        cls: Class to register.
        name: Optional unique identifier. Defaults to cls.__name__.
    """
    if category not in _REGISTRY:
        _REGISTRY[category] = {}
    identifier = name or cls.__name__
    _REGISTRY[category][identifier] = cls


def register(category: str, name: Optional[str] = None) -> Callable[[Type[Any]], Type[Any]]:
    """Optional decorator that registers a class under a functional domain or custom alias.

    Note: Subclassing AIMLite base classes (Model, Dataset, BaseTrainer, etc.)
    already registers the subclass automatically under the hood. This decorator
    is only needed if you wish to define an explicit custom alias.

    Args:
        category: Functional domain (e.g. "model", "dataset", "trainer", "evaluator", "inference", "config").
        name: Optional unique identifier string. If None, defaults to cls.__name__.

    Returns:
        Decorator callable returning the original class reference.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        register_class(category, cls, name=name)
        return cls

    return decorator


def get(category: str, name: str) -> Type[Any]:
    """Looks up and returns the registered class reference by name.

    Supports exact matching first, then falls back to case-insensitive matching.

    Args:
        category: Functional domain category.
        name: Registered identifier name.

    Returns:
        The registered class reference.

    Raises:
        KeyError: If the requested category or identifier has not been registered.
    """
    if category not in _REGISTRY:
        raise KeyError(
            f"No registered items for category '{category}'. Available categories: {list(_REGISTRY.keys())}"
        )

    cat_items = _REGISTRY[category]
    if name in cat_items:
        return cat_items[name]

    # Case-insensitive fallback
    name_lower = name.lower()
    for key, cls in cat_items.items():
        if key.lower() == name_lower:
            return cls

    available = list(cat_items.keys())
    raise KeyError(
        f"No registered item found for category '{category}' with name '{name}'. "
        f"Available items in '{category}': {available}"
    )


def get_all(category: str) -> Dict[str, Type[Any]]:
    """Returns a dictionary of all registered classes for a given category.

    Args:
        category: Functional domain category (e.g. "model", "dataset").

    Returns:
        Dictionary mapping class names to class types.
    """
    return dict(_REGISTRY.get(category, {}))


def clear_registry(category: Optional[str] = None) -> None:
    """Utility helper to clear registrations (primarily for testing)."""
    if category is not None:
        _REGISTRY.pop(category, None)
    else:
        _REGISTRY.clear()

