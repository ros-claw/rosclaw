import copy
from dataclasses import replace

import numpy as np
import pytest

from rosclaw.growth import bounded_advantage_regression as legacy
from rosclaw.growth import proposal_advantage_regression as proposal
from rosclaw.growth.correlated_residual_gradient import conditional_means
from tests.growth.test_bounded_advantage_regression import batch


def inputs(*, budget=0.005, large_actions=False):
    data = batch()
    data["config"] = proposal.ProposalAdvantageRegressionConfig(
        steps=160 if large_actions else 40, maximum_mean_kl=budget
    )
    if large_actions:
        data["actions"] *= 7.5
        mu = conditional_means(data["baseline"], data["actions"], data["first"], 0.9)
        sigma = data["marginal_std"] * np.where(data["first"], 1, np.sqrt(1 - 0.9**2))
        data["old_log_probability"] = (
            -0.5 * ((data["actions"] - mu) / sigma[:, None]) ** 2
            - np.log(sigma[:, None])
            - 0.5 * np.log(2 * np.pi)
        ).sum(1)
    return data


def test_default_numerics_match_legacy_without_changing_legacy_api():
    pytest.importorskip("torch")
    old = legacy.fit_advantage_residual(**batch())
    new = proposal.fit_proposal_advantage_residual(**inputs())
    new_only = {"execution_ceiling", "maximum_mean_kl", "runtime_execution_authorized"}
    assert {k: v for k, v in old.items() if k != "algorithm"} == {
        k: v for k, v in new.items() if k != "algorithm" and k not in new_only
    }
    assert new["algorithm"] == "PROPOSAL_TRUST_REGION_ADVANTAGE_REGRESSION_V1"
    assert new["execution_ceiling"] == "PROPOSAL_ONLY_NO_RUNTIME"
    assert old["algorithm"] == "BOUNDED_ADVANTAGE_WEIGHTED_RESIDUAL_REGRESSION_V1"


def test_larger_explicit_budget_remains_proposal_only_and_preserves_guard_and_inputs():
    pytest.importorskip("torch")
    data = inputs(budget=0.25, large_actions=True)
    original = copy.deepcopy(data)
    learned = proposal.fit_proposal_advantage_residual(**data)
    hidden = data["context"]
    for layer in learned["layers"]:
        hidden = np.tanh(hidden @ np.array(layer["weight"]).T + np.array(layer["bias"]))
    mean = data["baseline"] + 0.2 * data["gates"][:, None] * hidden
    assert mean[0, 0] == 0
    assert learned["exact_mean_marginal_kl"] > 0.005
    assert max(learned["exact_mean_conditional_kl"], learned["exact_mean_marginal_kl"]) <= 0.25
    assert learned["full_batch_loss_history"][-1] < learned["full_batch_loss_history"][0]
    for key in (
        "physical_batch_verified",
        "distributional_retention_guaranteed",
        "runtime_execution_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert learned[key] is False
    for key in ("context", "baseline", "gates", "actions", "old_log_probability"):
        np.testing.assert_array_equal(data[key], original[key])
    for (w, b), (ow, ob) in zip(data["layers"], original["layers"], strict=True):
        np.testing.assert_array_equal(w, ow)
        np.testing.assert_array_equal(b, ob)


@pytest.mark.parametrize("budget", [True, float("nan"), float("inf"), 0, 0.004, 0.501, 1])
def test_invalid_proposal_budget_fails_closed(budget):
    with pytest.raises(ValueError, match="proposal-only"):
        replace(proposal.ProposalAdvantageRegressionConfig(), maximum_mean_kl=budget).validate()


@pytest.mark.parametrize("ceiling", ["SIM_ONLY", "SHADOW", "REAL", None])
def test_this_numeric_api_cannot_grant_runtime_execution(ceiling):
    with pytest.raises(ValueError, match="proposal-only"):
        replace(proposal.ProposalAdvantageRegressionConfig(), execution_ceiling=ceiling).validate()


@pytest.mark.parametrize("fault", ["density", "nan", "gate", "reset", "legacy-config"])
def test_wider_budget_does_not_bypass_behavior_or_shape_checks(fault):
    data = inputs(budget=0.25)
    if fault == "density":
        data["old_log_probability"][0] += 0.01
    elif fault == "nan":
        data["context"][0, 0] = float("nan")
    elif fault == "gate":
        data["gates"][0] = -1
    elif fault == "reset":
        data["first"][0] = False
    else:
        data["config"] = legacy.AdvantageRegressionConfig()
    with pytest.raises(ValueError):
        proposal.fit_proposal_advantage_residual(**data)


def test_host_torch_settings_and_rng_restored_on_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail(*args, **kwargs):
        raise RuntimeError("proposal fixture failure")

    monkeypatch.setattr(proposal, "_fit", fail)
    with pytest.raises(RuntimeError, match="proposal fixture failure"):
        proposal.fit_proposal_advantage_residual(**inputs(budget=0.25))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
