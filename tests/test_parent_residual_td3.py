"""Numeric unit evidence only, never robot behavior or physical provenance."""

import copy
import io

import numpy as np
import pytest

from rosclaw.growth.parent_residual_td3 import ParentResidualTD3Config, fit_parent_residual_td3


def bank():
    rng = np.random.default_rng(42)
    context = rng.normal(size=(96, 4)).astype(np.float32)
    features = np.concatenate((context, np.ones((96, 1), np.float32)), axis=1)
    actions = rng.uniform(-1, 1, size=(96, 2)).astype(np.float32)
    reward = -np.square(actions - np.tanh(context[:, :2]) * 0.3).sum(axis=1)
    return {
        "context": context,
        "features": features,
        "actions": actions,
        "rewards": reward,
        "next_context": context.copy(),
        "next_features": features.copy(),
        "terminated": np.ones(96, np.bool_),
        "actor_weight": np.zeros((2, 5), np.float32),
        "actor_bias": np.zeros(2, np.float32),
        "context_hash": "sha256:" + "a" * 64,
    }


def torch_module():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    return torch


def test_critic_only_warmup_keeps_zero_parent_and_caller_arrays_and_rng():
    torch = torch_module()
    data = bank()
    before = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    rng = torch.get_rng_state().clone()
    result = fit_parent_residual_td3(
        **data, config=ParentResidualTD3Config(critic_steps=8, critic_warmup_steps=8, batch_size=16)
    )
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
    np.testing.assert_array_equal(result["actor_weight"], data["actor_weight"])
    assert result["report"]["actual_actor_optimizer_updates"] == 0
    assert result["report"]["actual_critic_optimizer_updates"] == 8
    assert not result["report"]["physical_task_gain_proven"]
    assert not result["report"]["hardware_authorized"]


def test_delayed_actor_and_twin_critics_actually_update_and_resume_safely():
    torch = torch_module()
    data = bank()
    config = ParentResidualTD3Config(critic_steps=32, critic_warmup_steps=16, batch_size=16)
    first = fit_parent_residual_td3(**data, config=config)
    assert first["report"]["actor_attempts"] == 8
    assert first["report"]["actual_actor_optimizer_updates"] > 0
    assert np.any(first["actor_weight"])
    saved = io.BytesIO()
    torch.save(first["checkpoint"], saved)
    saved.seek(0)
    loaded = torch.load(saved, map_location="cpu", weights_only=True)
    data["actor_weight"], data["actor_bias"] = first["actor_weight"], first["actor_bias"]
    second = fit_parent_residual_td3(**data, config=config, checkpoint=loaded)
    assert second["report"]["total_critic_optimizer_updates"] == 64
    assert (
        second["report"]["total_actor_optimizer_updates"]
        > first["report"]["total_actor_optimizer_updates"]
    )
    repeat = fit_parent_residual_td3(**data, config=config, checkpoint=loaded)
    np.testing.assert_array_equal(second["actor_weight"], repeat["actor_weight"])
    assert second["report"] == repeat["report"]


def test_rejected_actor_step_rolls_back_adam_not_just_weights():
    torch_module()
    config = ParentResidualTD3Config(
        critic_steps=32,
        critic_warmup_steps=16,
        batch_size=16,
        actor_learning_rate=0.01,
        maximum_action_rms=0.001,
    )
    result = fit_parent_residual_td3(**bank(), config=config)
    assert result["report"]["actor_trust_rejections"] == 8
    assert result["report"]["actual_actor_optimizer_updates"] == 0
    assert result["checkpoint"]["actor_optimizer"]["state"] == {}
    np.testing.assert_array_equal(result["actor_weight"], np.zeros((2, 5)))


@pytest.mark.parametrize("fault", ["nan", "action", "integer_overflow", "bool_done", "dimension"])
def test_bad_transition_bank_is_rejected_before_optimizer(fault):
    data = bank()
    if fault == "nan":
        data["features"][0, 0] = np.nan
    elif fault == "action":
        data["actions"][0, 0] = 2
    elif fault == "integer_overflow":
        data["context"] = np.full((96, 4), np.iinfo(np.int64).min)
    elif fault == "bool_done":
        data["terminated"] = np.zeros(96, np.float32)
    else:
        data["next_features"] = np.zeros((95, 5))
    with pytest.raises(ValueError):
        fit_parent_residual_td3(**data, config=ParentResidualTD3Config(critic_steps=1))


@pytest.mark.parametrize("fault", ["context", "actor", "counter", "moments", "lr"])
def test_checkpoint_cannot_rebind_context_or_reset_optimizer_history(fault):
    torch_module()
    data = bank()
    config = ParentResidualTD3Config(critic_steps=8, critic_warmup_steps=8, batch_size=16)
    first = fit_parent_residual_td3(**data, config=config)
    ckpt = copy.deepcopy(first["checkpoint"])
    if fault == "context":
        ckpt["context_hash"] = "sha256:" + "b" * 64
    elif fault == "actor":
        ckpt["actor"]["weight"][0, 0] = 0.2
    elif fault == "counter":
        ckpt["total_critic_updates"] = 0
    elif fault == "moments":
        ckpt["critic_optimizer"]["state"] = {}
    else:
        ckpt["critic_optimizer"]["param_groups"][0]["lr"] = 9.0
    with pytest.raises(ValueError):
        fit_parent_residual_td3(**data, config=config, checkpoint=ckpt)
