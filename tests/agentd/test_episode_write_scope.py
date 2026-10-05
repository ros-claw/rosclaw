"""Prevent a valid native logfile from encountering an undisclosed policy."""

import copy

import pytest

from rosclaw.agentd.episode_write_scope import (
    EpisodeWriteScopeError,
    validate_episode_write_scope,
)


def test_observed_native_log_path_disagreement_rejects_before_clock(tmp_path):
    public = ["native/ac_context_adapter.py", "native/source_tests.log"]
    operator = ["native/ac_context_adapter.py", "native/test_stdout.txt"]
    before = copy.deepcopy((public, operator))
    with pytest.raises(EpisodeWriteScopeError, match="SCOPE_MISMATCH"):
        validate_episode_write_scope(public, operator)
    assert (public, operator) == before
    assert list(tmp_path.iterdir()) == []


def test_explicit_logs_reports_and_source_paths_match_without_mutation():
    public = ["native/solver.py", "native/test_stdout.txt", "reports/source.json"]
    operator = list(reversed(public))
    before = copy.deepcopy((public, operator))
    result = validate_episode_write_scope(
        public, operator, protected_inputs=["inputs/full608.json"]
    )
    assert result == frozenset(public)
    assert (public, operator) == before


@pytest.mark.parametrize(
    "path",
    ["", ".", "../outside", "a/../b", "/tmp/log", "a//b", "./log", "a\\b", "log\x00"],
)
def test_noncanonical_declarations_cannot_hide_scope_disagreement(path):
    with pytest.raises(EpisodeWriteScopeError):
        validate_episode_write_scope([path], [path])


@pytest.mark.parametrize("paths", ["reports/logs", [True], ["log", "log"]])
def test_prose_wrong_types_and_duplicate_declarations_are_not_path_contracts(paths):
    with pytest.raises(EpisodeWriteScopeError):
        validate_episode_write_scope(paths, paths)


def test_public_operator_agreement_does_not_make_inputs_writable():
    with pytest.raises(EpisodeWriteScopeError, match="PROTECTED_INPUT"):
        validate_episode_write_scope(
            ["inputs/full608.json"],
            ["inputs/full608.json"],
            protected_inputs=["inputs/full608.json"],
        )


def test_readonly_episode_can_declare_no_output_paths():
    assert validate_episode_write_scope([], []) == frozenset()
