from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import pytest

from rosclaw import cli_extensions


@dataclass
class _EntryPoint:
    name: str
    value: Any
    error: Exception | None = None

    def load(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


class _EntryPoints(list[_EntryPoint]):
    def select(self, *, group: str) -> _EntryPoints:
        assert group == cli_extensions.CLI_EXTENSION_GROUP
        return self


def _parsers() -> tuple[argparse.ArgumentParser, Any]:
    parser = argparse.ArgumentParser()
    return parser, parser.add_subparsers(dest="command")


def test_register_and_dispatch_downstream_command(monkeypatch: pytest.MonkeyPatch) -> None:
    parser, subparsers = _parsers()

    def register(target: Any) -> None:
        sample = target.add_parser("sample")
        sample.set_defaults(rosclaw_extension_handler=lambda _args: 7)

    monkeypatch.setattr(
        cli_extensions.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_EntryPoint("sample", register)]),
    )

    report = cli_extensions.register_cli_extensions(subparsers)
    args = parser.parse_args(["sample"])

    assert report.loaded == ("sample",)
    assert report.errors == ()
    assert cli_extensions.dispatch_cli_extension(args) == 7


def test_broken_extension_does_not_hide_core_commands(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    parser, subparsers = _parsers()
    subparsers.add_parser("core")
    monkeypatch.setattr(
        cli_extensions.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_EntryPoint("broken", None, RuntimeError("boom"))]),
    )

    report = cli_extensions.register_cli_extensions(subparsers)

    assert parser.parse_args(["core"]).command == "core"
    assert report.loaded == ()
    assert report.errors == ("broken: boom",)
    assert "broken: boom" in caplog.text


def test_extensions_can_be_disabled_for_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    _parser, subparsers = _parsers()
    monkeypatch.setenv("ROSCLAW_DISABLE_CLI_EXTENSIONS", "true")
    monkeypatch.setattr(
        cli_extensions.importlib.metadata,
        "entry_points",
        lambda: pytest.fail("disabled discovery must not inspect entry points"),
    )

    report = cli_extensions.register_cli_extensions(subparsers)

    assert report.disabled is True
    assert report.loaded == ()


def test_non_callable_handler_is_rejected() -> None:
    args = argparse.Namespace(rosclaw_extension_handler="unsafe")
    with pytest.raises(TypeError, match="must be callable"):
        cli_extensions.dispatch_cli_extension(args)
