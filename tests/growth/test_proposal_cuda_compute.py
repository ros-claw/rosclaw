"""Numeric CUDA computation only; never robot or physical-policy authority."""

from dataclasses import replace

import numpy as np
import pytest

from rosclaw.growth import proposal_advantage_regression as proposal
from tests.growth.test_proposal_advantage_regression import inputs


@pytest.mark.parametrize("device", [None, True, "cuda", "cuda:-1", "cuda:100", "REAL", "mps"])
def test_invalid_device_fails_without_torch_import_or_optimizer(device):
    with pytest.raises(ValueError, match="compute device"):
        replace(proposal.ProposalAdvantageRegressionConfig(), compute_device=device).validate()


def test_gpu_requires_explicit_deterministic_cublas_environment(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    data = inputs(budget=0.05)
    data["config"] = replace(data["config"], compute_device="cuda:0")
    with pytest.raises(ValueError, match="cuBLAS"):
        proposal.fit_proposal_advantage_residual(**data)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_actual_cuda_fit_cpu_numeric_comparison_and_rng_isolation(index, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= index:
        pytest.skip("actual requested CUDA device is unavailable")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    data = inputs(budget=0.05)
    cpu = proposal.fit_proposal_advantage_residual(**data)
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(index).clone()
    current = torch.cuda.current_device()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    gpu_data = dict(data, config=replace(data["config"], compute_device=f"cuda:{index}"))
    gpu = proposal.fit_proposal_advantage_residual(**gpu_data)
    assert gpu["compute_device"] == f"cuda:{index}"
    assert gpu["cross_device_bit_identity_claimed"] is False
    assert gpu["completed_optimizer_steps"] == cpu["completed_optimizer_steps"]
    for old, new in zip(cpu["layers"], gpu["layers"], strict=True):
        for field in ("weight", "bias"):
            np.testing.assert_allclose(new[field], old[field], rtol=0, atol=1e-10)
    for field in ("full_batch_loss_history", "exact_mean_conditional_kl", "exact_mean_marginal_kl"):
        np.testing.assert_allclose(gpu[field], cpu[field], rtol=0, atol=1e-10)
    assert gpu["hardware_authorized"] is gpu["runtime_execution_authorized"] is False
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(index))
    assert torch.cuda.current_device() == current
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_cuda_rng_and_settings_restore_after_optimizer_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(0).clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA fit fixture failed")

    monkeypatch.setattr(proposal, "_fit", fail)
    data = inputs(budget=0.05)
    data["config"] = replace(data["config"], compute_device="cuda:0")
    with pytest.raises(RuntimeError, match="CUDA fit fixture"):
        proposal.fit_proposal_advantage_residual(**data)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
