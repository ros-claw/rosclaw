"""Regression for copied primary-source lists and partial manifest closures."""

import hashlib
from pathlib import Path

import pytest

from rosclaw.agentd.episode_source_bindings import (
    EpisodeSourceBindingsError,
    validate_episode_source_bindings,
)


def bundle(tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    sources = {}
    inputs = {}
    for name, target in [
        ("native/adapter.py", sources),
        ("native/solver.py", sources),
        ("tests/test_adapter.py", sources),
        ("inputs/context.json", inputs),
        ("inputs/parent.py", inputs),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# Protocol fixture, no application implementation.\n")
        target[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources, inputs


def test_new_primary_sources_and_complete_inputs_admitted_without_writes(tmp_path: Path):
    sources, inputs = bundle(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert validate_episode_source_bindings(tmp_path, sources, list(sources), inputs, inputs) == {
        **sources,
        **inputs,
    }
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_old_copied_primary_list_rejects_new_bundle(tmp_path: Path):
    sources, inputs = bundle(tmp_path)
    with pytest.raises(EpisodeSourceBindingsError, match="PRIMARY_BINDINGS_MISSING"):
        validate_episode_source_bindings(
            tmp_path, sources, ["native/old_solver.py"], inputs, inputs
        )
    for omitted in sources:
        with pytest.raises(EpisodeSourceBindingsError, match="PRIMARY_BINDINGS_MISSING"):
            validate_episode_source_bindings(
                tmp_path,
                {p: h for p, h in sources.items() if p != omitted},
                list(sources),
                inputs,
                inputs,
            )


def test_self_consistent_partial_or_changed_input_closure_rejected(tmp_path: Path):
    sources, inputs = bundle(tmp_path)
    partial = {"inputs/context.json": inputs["inputs/context.json"]}
    with pytest.raises(EpisodeSourceBindingsError, match="PROTECTED_CLOSURE_MISMATCH"):
        validate_episode_source_bindings(tmp_path, sources, list(sources), partial, inputs)
    changed = {**inputs, "inputs/parent.py": "f" * 64}
    with pytest.raises(EpisodeSourceBindingsError, match="PROTECTED_CLOSURE_MISMATCH"):
        validate_episode_source_bindings(tmp_path, sources, list(sources), changed, inputs)


def test_declared_hash_mutation_and_path_escape_rejected(tmp_path: Path):
    sources, inputs = bundle(tmp_path)
    (tmp_path / "native/adapter.py").write_bytes(b"# Changed after manifest.\n")
    with pytest.raises(EpisodeSourceBindingsError, match="FILE_HASH_MISMATCH"):
        validate_episode_source_bindings(tmp_path, sources, list(sources), inputs, inputs)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(b"external fixture")
    alias = tmp_path / "alias.py"
    alias.symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(EpisodeSourceBindingsError, match="FILE_OUTSIDE_OR_INVALID"):
        validate_episode_source_bindings(
            tmp_path, {"alias.py": digest}, ["alias.py"], inputs, inputs
        )
    with pytest.raises(EpisodeSourceBindingsError, match="PATH_NOT_CANONICAL"):
        validate_episode_source_bindings(
            tmp_path, {"../outside": digest}, ["../outside"], inputs, inputs
        )
