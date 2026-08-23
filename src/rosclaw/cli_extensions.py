"""Discover downstream command trees without coupling them to ROSClaw core."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("rosclaw.cli_extensions")

CLI_EXTENSION_GROUP = "rosclaw.cli_extensions"
CLI_EXTENSION_HANDLER = "rosclaw_extension_handler"
_DISABLE_ENV = "ROSCLAW_DISABLE_CLI_EXTENSIONS"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

CLIExtensionRegister = Callable[[Any], None]
CLIExtensionHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True)
class CLIExtensionReport:
    """Immutable discovery result for diagnostics and tests."""

    loaded: tuple[str, ...]
    errors: tuple[str, ...]
    disabled: bool = False


def register_cli_extensions(
    subparsers: Any,
    *,
    entry_point_group: str = CLI_EXTENSION_GROUP,
) -> CLIExtensionReport:
    """Load installed downstream CLI adapters, failing open for core commands.

    An adapter must expose ``register_cli(subparsers)`` through the
    ``rosclaw.cli_extensions`` entry-point group. Adapters may add command
    parsers, but they do not receive a runtime, driver, or hardware handle.
    Set ``ROSCLAW_DISABLE_CLI_EXTENSIONS=1`` to recover from a broken optional
    installation.
    """

    if os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUE_VALUES:
        return CLIExtensionReport(loaded=(), errors=(), disabled=True)

    try:
        entry_points = importlib.metadata.entry_points()
        selected: Iterable[Any]
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=entry_point_group)
        else:  # pragma: no cover - compatibility with older metadata providers
            selected = entry_points.get(entry_point_group, ())
    except Exception as exc:
        message = f"discovery failed: {exc}"
        logger.warning("ROSClaw CLI extension %s", message)
        return CLIExtensionReport(loaded=(), errors=(message,))

    loaded: list[str] = []
    errors: list[str] = []
    ordered = sorted(
        selected,
        key=lambda item: (
            str(getattr(item, "name", "")),
            str(getattr(item, "value", "")),
        ),
    )
    for entry_point in ordered:
        name = str(getattr(entry_point, "name", getattr(entry_point, "value", "unknown")))
        try:
            register = entry_point.load()
            if not callable(register):
                raise TypeError("entry point must resolve to a callable")
            register(subparsers)
        except Exception as exc:
            message = f"{name}: {exc}"
            errors.append(message)
            logger.warning("Failed to register ROSClaw CLI extension %s", message)
            continue
        loaded.append(name)
        logger.debug("Registered ROSClaw CLI extension: %s", name)

    return CLIExtensionReport(loaded=tuple(loaded), errors=tuple(errors))


def dispatch_cli_extension(args: argparse.Namespace) -> int | None:
    """Run an extension handler selected by argparse, if present."""

    handler = getattr(args, CLI_EXTENSION_HANDLER, None)
    if handler is None:
        return None
    if not callable(handler):
        raise TypeError(f"{CLI_EXTENSION_HANDLER} must be callable")
    return int(handler(args))


__all__ = [
    "CLI_EXTENSION_GROUP",
    "CLI_EXTENSION_HANDLER",
    "CLIExtensionReport",
    "dispatch_cli_extension",
    "register_cli_extensions",
]
