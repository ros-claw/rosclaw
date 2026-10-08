import copy
from dataclasses import fields, replace

import numpy as np
import pytest

from rosclaw.growth.extended_proposal_regression import (
    ExtendedProposalRegressionConfig,
    fit_extended_proposal_residual,
)
from rosclaw.growth.proposal_advantage_regression import (
    ProposalAdvantageRegressionConfig,
    fit_proposal_advantage_residual,
)
from tests.growth.test_proposal_advantage_regression import inputs


def extended(config):
    return ExtendedProposalRegressionConfig(
        **{f.name: getattr(config, f.name) for f in fields(ProposalAdvantageRegressionConfig)}
    )


def test_same_budget_preserves_all_original_numerics_and_inputs():
    pytest.importorskip("torch")
    data = inputs(budget=0.05)
    before = copy.deepcopy(data)
    original = fit_proposal_advantage_residual(**data)
    data["config"] = extended(data["config"])
    result = fit_extended_proposal_residual(**data)
    assert result["algorithm"] == "EXPLICIT_EXTENDED_PROPOSAL_REGRESSION_V1"
    assert result["base_numeric_algorithm"] == original["algorithm"]
    assert all(result[k] == v for k, v in original.items() if k != "algorithm")
    assert result["additional_physical_episodes_executed"] == 0
    assert result["kl_budget_reset_between_passes"] is False
    for name in ("context", "baseline", "gates", "actions", "old_log_probability"):
        assert np.array_equal(data[name], before[name])


def test_actual_longer_optimization_still_uses_one_original_kl_reference():
    pytest.importorskip("torch")
    data = inputs(budget=0.05, large_actions=True)
    data["config"] = replace(extended(data["config"]), steps=192, learning_rate=1e-6)
    result = fit_extended_proposal_residual(**data)
    assert result["completed_optimizer_steps"] == 192
    assert len(result["full_batch_loss_history"]) == 193
    assert max(result["exact_mean_marginal_kl"], result["exact_mean_conditional_kl"]) <= 0.05
    assert result["single_initial_behavior_reference"] is True
    for name in (
        "runtime_execution_authorized",
        "promotion_authorized",
        "hardware_authorized",
        "physical_batch_verified",
        "distributional_retention_guaranteed",
    ):
        assert result[name] is False


@pytest.mark.parametrize("steps", [True, 0, -1, 2561, 1600.0, None])
def test_extended_budget_fails_closed(steps):
    with pytest.raises(ValueError):
        replace(ExtendedProposalRegressionConfig(), steps=steps).validate()


def test_original_budget_and_density_checks_remain_intact():
    with pytest.raises(ValueError):
        ProposalAdvantageRegressionConfig(steps=161).validate()
    data = inputs()
    with pytest.raises(ValueError):
        fit_extended_proposal_residual(**data)
    data["config"] = replace(extended(data["config"]), steps=200)
    data["old_log_probability"][0] += 0.01
    with pytest.raises(ValueError, match="likelihood"):
        fit_extended_proposal_residual(**data)


def test_host_torch_state_restored_when_extended_solver_fails(monkeypatch):
    torch = pytest.importorskip("torch")
    from rosclaw.growth import proposal_advantage_regression as original

    rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail(*args, **kwargs):
        raise RuntimeError("extended fixture failure")

    monkeypatch.setattr(original, "_fit", fail)
    data = inputs()
    data["config"] = replace(extended(data["config"]), steps=1600)
    with pytest.raises(RuntimeError, match="extended fixture"):
        fit_extended_proposal_residual(**data)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


@pytest.mark.parametrize(
    "field,value",
    [
        ("residual_cap", 0.201),
        ("maximum_mean_kl", 0.501),
        ("execution_ceiling", "REAL"),
        ("rho", 0.96),
        ("compute_device", "cuda:-1"),
    ],
)
def test_all_other_original_limits_are_preserved(field, value):
    with pytest.raises(ValueError):
        replace(ExtendedProposalRegressionConfig(), **{field: value}).validate()
