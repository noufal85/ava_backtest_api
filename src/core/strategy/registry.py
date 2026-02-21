"""Strategy registry — maps name → class."""
from src.core.strategy.base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[BaseStrategy]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_strategies() -> list[dict]:
    return [
        {
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
            "category": cls.category,
            "tags": cls.tags,
        }
        for cls in _REGISTRY.values()
    ]
