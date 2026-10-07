"""Deterministic evidence-first ROS diagnostics."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import diagnose


def __getattr__(name):
    # Passive audit tools are standard-library-only in the ROS container.
    # Loading them must not import the system-model/Pydantic diagnosis engine.
    if name == "diagnose":
        from .engine import diagnose

        return diagnose
    raise AttributeError(name)


__all__ = ["diagnose"]
