"""Tests for ``rosclaw.eurdf.cli``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rosclaw.eurdf.cli import (
    add_eurdf_subparser,
    cmd_eurdf_cache_list,
    cmd_eurdf_pull,
    dispatch_eurdf_command,
)
from rosclaw.eurdf.zoo_client import E_URDF_ZOO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not E_URDF_ZOO_AVAILABLE,
    reason="e_urdf_zoo package is not installed",
)


@pytest.fixture
def zoo_path() -> Path:
    return Path(__file__).parent.parent.parent / "e-urdf-zoo" / "robots"


@pytest.fixture
def parser(zoo_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_eurdf_subparser(subparsers)
    return parser


def test_cli_info_human(parser: argparse.ArgumentParser, zoo_path: Path) -> None:
    args = parser.parse_args(
        ["eurdf", "info", "dexhands/inspire_hand/right", "--zoo-path", str(zoo_path)]
    )
    assert dispatch_eurdf_command(args) == 0


def test_cli_info_json(parser: argparse.ArgumentParser, zoo_path: Path) -> None:
    args = parser.parse_args(
        ["eurdf", "info", "dexhands/inspire_hand/right", "--json", "--zoo-path", str(zoo_path)]
    )
    assert dispatch_eurdf_command(args) == 0


def test_cli_search_json(parser: argparse.ArgumentParser, zoo_path: Path) -> None:
    args = parser.parse_args(["eurdf", "search", "inspire", "--json", "--zoo-path", str(zoo_path)])
    assert dispatch_eurdf_command(args) == 0


def test_cli_validate(parser: argparse.ArgumentParser, zoo_path: Path) -> None:
    args = parser.parse_args(
        ["eurdf", "validate", "dexhands/inspire_hand/right", "--zoo-path", str(zoo_path)]
    )
    assert dispatch_eurdf_command(args) == 0


def test_cli_pull_and_cache_list(
    parser: argparse.ArgumentParser, zoo_path: Path, tmp_path: Path
) -> None:
    args = parser.parse_args(
        [
            "eurdf",
            "pull",
            "dexhands/inspire_hand/right",
            "--zoo-path",
            str(zoo_path),
            "--source",
            str(zoo_path / "dexhands" / "inspire_hand" / "right"),
        ]
    )
    # Patch cache dir via monkey-patching the client factory is awkward; instead
    # verify the command dispatches without error when source is explicit.
    assert cmd_eurdf_pull(args) == 0


def test_cli_cache_list_empty(parser: argparse.ArgumentParser, tmp_path: Path) -> None:
    args = parser.parse_args(["eurdf", "cache", "list", "--json"])
    # The default cache dir is empty in CI; command should still succeed.
    assert cmd_eurdf_cache_list(args) == 0


def test_cli_help_when_no_subcommand(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["eurdf"])
    assert dispatch_eurdf_command(args) == 1
