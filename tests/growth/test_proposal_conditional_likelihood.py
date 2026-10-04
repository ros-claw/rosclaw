"""Independent conditional objective checks; no simulator or policy promotion."""

from dataclasses import replace

import numpy as np
import pytest

from rosclaw.growth import proposal_advantage_regression as proposal
from tests.growth.test_proposal_advantage_regression import inputs


def conditional_data(rho=0.9):
    data = inputs(budget=0.05)
    data["config"] = replace(data["config"], rho=rho, likelihood_profile="conditional-ar1")
    conditional = data["baseline"].copy()
    for i in range(len(conditional)):
        if not data["first"][i]:
            conditional[i] += rho * (data["actions"][i - 1] - data["baseline"][i - 1])
    sigma = data["marginal_std"] * np.where(data["first"], 1, np.sqrt(1 - rho**2))
    data["old_log_probability"] = (
        -0.5 * ((data["actions"] - conditional) / sigma[:, None]) ** 2
        - np.log(sigma[:, None])
        - 0.5 * np.log(2 * np.pi)
    ).sum(axis=1)
    return data


def test_conditional_loss_recomputed_with_recorded_previous_actions_and_reset_rows():
    pytest.importorskip("torch")
    data = conditional_data()
    learned = proposal.fit_proposal_advantage_residual(**data)
    hidden = data["context"]
    for layer in learned["layers"]:
        hidden = np.tanh(hidden @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
    mean = data["baseline"] + 0.2 * data["gates"][:, None] * hidden
    conditional, original = mean.copy(), data["baseline"].copy()
    for i in range(len(mean)):
        if not data["first"][i]:
            conditional[i] += 0.9 * (data["actions"][i - 1] - mean[i - 1])
            original[i] += 0.9 * (data["actions"][i - 1] - data["baseline"][i - 1])
    noise = data["marginal_std"] * np.where(data["first"], 1, np.sqrt(1 - 0.9**2))
    ckl = np.mean(np.sum((conditional - original) ** 2 / (2 * noise[:, None] ** 2), axis=1))
    weights = np.exp(np.clip(data["advantages"] / 0.5, -64, np.log(20)))
    weights /= weights.mean()
    loss = np.mean(weights * (0.5 * ((data["actions"] - conditional) / noise[:, None]) ** 2).sum(1))
    np.testing.assert_allclose(
        learned["full_batch_loss_history"][-1], loss + 10 * ckl, rtol=0, atol=1e-10
    )
    assert learned["likelihood_profile"] == "conditional-ar1"
    assert learned["completed_optimizer_steps"] > 0
    assert mean[0, 0] == 0
    assert max(learned["exact_mean_conditional_kl"], learned["exact_mean_marginal_kl"]) <= 0.05
    assert learned["runtime_execution_authorized"] is learned["hardware_authorized"] is False


def test_zero_correlation_has_exact_original_marginal_numerics():
    pytest.importorskip("torch")
    data = conditional_data(rho=0.0)
    conditional = proposal.fit_proposal_advantage_residual(**data)
    marginal = proposal.fit_proposal_advantage_residual(
        **dict(data, config=replace(data["config"], likelihood_profile="marginal"))
    )
    assert {k: v for k, v in conditional.items() if k != "likelihood_profile"} == marginal


def test_conditional_actual_cuda_matches_cpu_without_claiming_bit_identity(monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("actual CUDA device 1 unavailable")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    data = conditional_data()
    cpu = proposal.fit_proposal_advantage_residual(**data)
    gpu = proposal.fit_proposal_advantage_residual(
        **dict(data, config=replace(data["config"], compute_device="cuda:1"))
    )
    for a, b in zip(cpu["layers"], gpu["layers"], strict=True):
        for key in ("weight", "bias"):
            np.testing.assert_allclose(a[key], b[key], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        cpu["full_batch_loss_history"], gpu["full_batch_loss_history"], atol=1e-10, rtol=0
    )
    assert gpu["likelihood_profile"] == "conditional-ar1"
    assert gpu["cross_device_bit_identity_claimed"] is False


@pytest.mark.parametrize("profile", [None, True, "", "PPO", "iid", ["conditional-ar1"]])
def test_invalid_likelihood_profile_rejected(profile):
    with pytest.raises(ValueError, match="regression likelihood"):
        replace(proposal.ProposalAdvantageRegressionConfig(), likelihood_profile=profile).validate()
