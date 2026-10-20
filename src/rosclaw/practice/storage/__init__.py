"""Practice storage exports, loaded only when explicitly requested.

Catalog/layout users do not implicitly request the fallback HTTP client.
Explicit FallbackSync and star imports retain its genuine dependency effects.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.practice.storage.catalog import PracticeCatalog
    from rosclaw.practice.storage.fallback_sync import FallbackSync
    from rosclaw.practice.storage.layout import PracticeLayout

_EXPORT_MODULES = {
    "PracticeCatalog": "rosclaw.practice.storage.catalog",
    "FallbackSync": "rosclaw.practice.storage.fallback_sync",
    "PracticeLayout": "rosclaw.practice.storage.layout",
}

__all__ = ["PracticeCatalog", "FallbackSync", "PracticeLayout"]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
