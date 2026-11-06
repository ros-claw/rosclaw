from dataclasses import FrozenInstanceError

import pytest

from rosclaw.continual.counterfactual_binding import (
    CounterfactualKey,
    CoverageStatus,
    ObservedBranch,
    resolve_selected_branch,
)


def h(char: str) -> str:
    return "sha256:" + char * 64


def test_only_exact_selected_action_has_an_outcome() -> None:
    baseline = CounterfactualKey(h("a"), h("b"), h("c"))
    chosen = CounterfactualKey(h("a"), h("b"), h("d"))
    observed = ObservedBranch(baseline, h("e"), h("f"))
    missing = resolve_selected_branch(chosen, (observed,))
    assert missing.status is CoverageStatus.UNKNOWN
    assert missing.branch is None
    assert resolve_selected_branch(baseline, (observed,)).branch == observed


@pytest.mark.parametrize("change", ["boundary_hash", "condition_hash", "action_hash"])
def test_neither_different_state_nor_model_nor_action_is_a_match(change: str) -> None:
    values = {"boundary_hash": h("a"), "condition_hash": h("b"), "action_hash": h("c")}
    observed = ObservedBranch(CounterfactualKey(**values), h("e"), h("f"))
    values[change] = h("d")
    assert (
        resolve_selected_branch(CounterfactualKey(**values), (observed,)).status
        is CoverageStatus.UNKNOWN
    )


def test_duplicate_branch_and_invalid_receipt_are_rejected() -> None:
    key = CounterfactualKey(h("a"), h("b"), h("c"))
    branch = ObservedBranch(key, h("d"), h("e"))
    with pytest.raises(ValueError, match="duplicate"):
        resolve_selected_branch(key, (branch, branch))
    with pytest.raises(ValueError):
        ObservedBranch(key, "unbound", h("e"))


def test_hash_and_type_validation_and_immutability() -> None:
    with pytest.raises(ValueError):
        CounterfactualKey("SHA256:" + "a" * 64, h("b"), h("c"))
    key = CounterfactualKey(h("a"), h("b"), h("c"))
    with pytest.raises(ValueError):
        resolve_selected_branch(key, ("not a branch",))  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        key.action_hash = h("d")  # type: ignore[misc]
