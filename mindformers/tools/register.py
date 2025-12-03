"""Simple registry helpers."""
from enum import Enum


class MindFormerModuleType(str, Enum):
    """Available registry module types."""
    CONFIG = "config"


class MindFormerRegister:
    """Minimal registry that stores registered classes."""

    _registry = {}

    @classmethod
    def register(cls, module_type: MindFormerModuleType, legacy=False, search_names=None):
        """Decorator storing a class reference in the registry."""
        def decorator(obj):
            key = (module_type.value, getattr(obj, "__name__", str(obj)))
            cls._registry[key] = {
                "object": obj,
                "legacy": legacy,
                "search_names": search_names,
            }
            return obj

        return decorator

