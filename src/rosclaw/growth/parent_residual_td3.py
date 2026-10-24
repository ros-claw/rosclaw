"""Optional numeric TD3 updates over retained features and RAW residual actions.

No simulator, transport, action execution, or physical-provenance verifier.
The caller owns context/reward/feature semantics and applied-action evidence.
PyTorch is imported only when fitting. A trust cap and zero-parent anchor are
additional engineering restrictions, not a stability/plasticity guarantee.

References: https://arxiv.org/abs/1802.09477 and
https://spinningup.openai.com/en/latest/algorithms/td3.html
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParentResidualTD3Config:
    critic_steps: int = 1024
    critic_warmup_steps: int = 256
    policy_delay: int = 2
    batch_size: int = 256
    gamma: float = 0.995
    polyak: float = 0.995
    critic_learning_rate: float = 0.0003
    actor_learning_rate: float = 0.0001
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    parent_anchor_weight: float = 0.1
    maximum_action_rms: float = 0.15
    seed: int = 0

    def validate(self) -> None:
        integers = (
            (self.critic_steps, 1, 4096),
            (self.critic_warmup_steps, 0, 4096),
            (self.policy_delay, 1, 16),
            (self.batch_size, 1, 4096),
            (self.seed, 0, 2**32 - 1),
        )
        scalars = (
            (self.gamma, 0.5, 0.9999),
            (self.polyak, 0.5, 0.9999),
            (self.critic_learning_rate, 1e-6, 0.01),
            (self.actor_learning_rate, 1e-6, 0.01),
            (self.target_noise_std, 0.0, 0.5),
            (self.target_noise_clip, 0.01, 1.0),
            (self.parent_anchor_weight, 0.001, 100.0),
            (self.maximum_action_rms, 0.001, 0.5),
        )
        if any(type(x) is not int or not lo <= x <= hi for x, lo, hi in integers) or any(
            type(x) not in (int, float) or not math.isfinite(x) or not lo <= x <= hi
            for x, lo, hi in scalars
        ):
            raise ValueError("bounded explicit numeric TD3 configuration required")


def fit_parent_residual_td3(
    *,
    context: np.ndarray,
    features: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_context: np.ndarray,
    next_features: np.ndarray,
    terminated: np.ndarray,
    actor_weight: np.ndarray,
    actor_bias: np.ndarray,
    context_hash: str,
    config: ParentResidualTD3Config,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update twin critics and a delayed tanh head on caller-authenticated data.

    `actions` are normalized [-1,1] proposals BEFORE the environment guard, not
    the subsequently projected/applied commands. Features are frozen owned
    encoder outputs; this fitter never changes that encoder. `terminated` is
    true only for actual task termination, not an external rollout truncation.
    The caller must provide a measured next state even for truncated samples.
    """
    if type(config) is not ParentResidualTD3Config:
        raise ValueError("typed TD3 configuration required")
    config.validate()
    if checkpoint is not None and type(checkpoint) is not dict:
        raise ValueError("typed persistent numeric checkpoint required")
    if (
        not isinstance(context_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", context_hash) is None
    ):
        raise ValueError("immutable caller context commitment required")
    arrays = [
        np.asarray(x) for x in (context, features, actions, rewards, next_context, next_features)
    ]
    s, f, a, r, ns, nf = arrays
    if s.ndim != 2 or not 1 <= len(s) <= 200000 or not 1 <= s.shape[1] <= 1024:
        raise ValueError("bounded finite replay context required")
    n = len(s)
    done = np.asarray(terminated)
    if (
        f.ndim != 2
        or f.shape[0] != n
        or not 1 <= f.shape[1] <= 1024
        or a.ndim != 2
        or a.shape[0] != n
        or not 1 <= a.shape[1] <= 128
        or r.shape != (n,)
        or ns.shape != s.shape
        or nf.shape != f.shape
        or done.shape != (n,)
        or done.dtype != np.bool_
        or any(
            x.dtype.kind not in "fiu"
            or not np.isfinite(x).all()
            or (abs(x.astype(np.float64)) > 1e5).any()
            for x in arrays
        )
        or (abs(a) > 1).any()
        or (abs(r) > 1000).any()
    ):
        raise ValueError(
            "aligned measured transitions and raw normalized residual actions required"
        )
    weight, bias = np.asarray(actor_weight), np.asarray(actor_bias)
    if (
        weight.shape != (a.shape[1], f.shape[1])
        or bias.shape != (a.shape[1],)
        or any(
            x.dtype.kind not in "fiu"
            or not np.isfinite(x).all()
            or (abs(x.astype(np.float64)) > 1e4).any()
            for x in (weight, bias)
        )
    ):
        raise ValueError("finite exact-shaped retained actor head required")
    import torch

    # CPU numeric fitting does not commandeer a GPU or leak global RNG changes.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(config.seed)
        return _fit(
            arrays=[np.array(x, np.float32, copy=True) for x in arrays],
            done=np.array(done, copy=True),
            weight=np.array(weight, np.float32, copy=True),
            bias=np.array(bias, np.float32, copy=True),
            context_hash=context_hash,
            config=config,
            checkpoint=checkpoint,
        )


def _fit(
    *,
    arrays: list[np.ndarray],
    done: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    context_hash: str,
    config: ParentResidualTD3Config,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    import torch
    from torch import nn

    s, f, a, r, ns, nf = (torch.from_numpy(x) for x in arrays)
    d = torch.from_numpy(done)
    actor = nn.Linear(f.shape[1], a.shape[1], device="cpu", dtype=torch.float32)
    with torch.no_grad():
        actor.weight.copy_(torch.from_numpy(weight))
        actor.bias.copy_(torch.from_numpy(bias))
    target_actor = copy.deepcopy(actor)

    def critic() -> Any:
        return nn.Sequential(
            nn.Linear(s.shape[1] + a.shape[1], 128, device="cpu", dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128, device="cpu", dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 1, device="cpu", dtype=torch.float32),
        )

    q1, q2 = critic(), critic()
    tq1, tq2 = copy.deepcopy(q1), copy.deepcopy(q2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()], lr=config.critic_learning_rate
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    total_actor = total_critic = 0
    shape_commitment = [s.shape[1], f.shape[1], a.shape[1]]
    if checkpoint is not None:
        if (
            checkpoint.get("schema") != "rosclaw.parent_residual_td3.checkpoint.v1"
            or checkpoint.get("context_hash") != context_hash
            or checkpoint.get("shapes") != shape_commitment
            or checkpoint.get("config") != asdict(config)
            or checkpoint.get("torch_version") != str(torch.__version__)
        ):
            raise ValueError("exact persistent TD3 context, configuration and dimensions required")
        for key, model in (
            ("actor", actor),
            ("target_actor", target_actor),
            ("q1", q1),
            ("q2", q2),
            ("target_q1", tq1),
            ("target_q2", tq2),
        ):
            model.load_state_dict(copy.deepcopy(checkpoint[key]), strict=True)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("finite resumed TD3 weights required")
        if not np.array_equal(actor.weight.detach().numpy(), weight) or not np.array_equal(
            actor.bias.detach().numpy(), bias
        ):
            raise ValueError("checkpoint actor differs from the deployed head")
        actor_optimizer.load_state_dict(copy.deepcopy(checkpoint["actor_optimizer"]))
        critic_optimizer.load_state_dict(copy.deepcopy(checkpoint["critic_optimizer"]))
        generator.set_state(checkpoint["generator_state"].clone())
        total_actor, total_critic = (
            checkpoint["total_actor_updates"],
            checkpoint["total_critic_updates"],
        )
        if any(type(x) is not int or not 0 <= x <= 10**9 for x in (total_actor, total_critic)):
            raise ValueError("valid persisted optimizer counters required")
        for optimizer, count in ((actor_optimizer, total_actor), (critic_optimizer, total_critic)):
            if count > 0 and len(optimizer.state) != sum(
                len(g["params"]) for g in optimizer.param_groups
            ):
                raise ValueError("complete persisted Adam moments required")
            if any(float(v["step"]) != count for v in optimizer.state.values()):
                raise ValueError("persisted Adam step counters differ from update history")
            for param, value in optimizer.state.items():
                if set(value) != {"step", "exp_avg", "exp_avg_sq"} or any(
                    value[k].shape != param.shape or not bool(torch.isfinite(value[k]).all())
                    for k in ("exp_avg", "exp_avg_sq")
                ):
                    raise ValueError("finite shape-matched Adam moments required")
        for optimizer, lr in (
            (actor_optimizer, config.actor_learning_rate),
            (critic_optimizer, config.critic_learning_rate),
        ):
            for group in optimizer.param_groups:
                if (
                    group["lr"] != lr
                    or group["betas"] != (0.9, 0.999)
                    or group["eps"] != 1e-8
                    or group["weight_decay"] != 0
                    or group["amsgrad"]
                    or group["maximize"]
                ):
                    raise ValueError("resumed optimizer differs from the committed learning rule")

    def rms() -> float:
        with torch.no_grad():
            return float(torch.tanh(actor(f)).square().mean().sqrt())

    if rms() > config.maximum_action_rms + 1e-7:
        raise ValueError("deployed actor already exceeds this replay-bank trust cap")
    accepted = attempts = rejected = 0
    critic_loss_first = critic_loss_last = 0.0
    for index in range(config.critic_steps):
        rows = torch.randint(len(s), (config.batch_size,), generator=generator)
        with torch.no_grad():
            noise = (
                torch.randn(
                    (len(rows), a.shape[1]), generator=generator, device="cpu", dtype=torch.float32
                )
                * config.target_noise_std
            ).clamp(-config.target_noise_clip, config.target_noise_clip)
            target_action = (torch.tanh(target_actor(nf[rows])) + noise).clamp(-1, 1)
            q_input = torch.cat((ns[rows], target_action), dim=1)
            target = r[rows] + config.gamma * (~d[rows]).float() * torch.minimum(
                tq1(q_input).squeeze(1), tq2(q_input).squeeze(1)
            )
        inputs = torch.cat((s[rows], a[rows]), dim=1)
        loss = (q1(inputs).squeeze(1) - target).square().mean() + (
            q2(inputs).squeeze(1) - target
        ).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite TD3 critic target/loss")
        critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [*q1.parameters(), *q2.parameters()], 10.0, error_if_nonfinite=True
        )
        critic_optimizer.step()
        critic_loss_last = float(loss.detach())
        if index == 0:
            critic_loss_first = critic_loss_last
        total_critic += 1
        if total_critic % config.policy_delay:
            continue
        if total_critic > config.critic_warmup_steps:
            attempts += 1
            before, optimizer_before = (
                copy.deepcopy(actor.state_dict()),
                copy.deepcopy(actor_optimizer.state_dict()),
            )
            for p in q1.parameters():
                p.requires_grad_(False)
            proposal = torch.tanh(actor(f[rows]))
            objective = (
                -q1(torch.cat((s[rows], proposal), dim=1)).mean()
                + config.parent_anchor_weight * proposal.square().mean()
            )
            actor_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
            actor_optimizer.step()
            for p in q1.parameters():
                p.requires_grad_(True)
            if rms() > config.maximum_action_rms + 1e-7:
                actor.load_state_dict(before)
                actor_optimizer.load_state_dict(optimizer_before)
                rejected += 1
            else:
                accepted += 1
                total_actor += 1
        # Delayed targets also advance during critic-only warm-up. The actor
        # remains exactly retained until the first accepted actor update.
        with torch.no_grad():
            for target_model, model in ((target_actor, actor), (tq1, q1), (tq2, q2)):
                for tp, p in zip(target_model.parameters(), model.parameters(), strict=True):
                    tp.mul_(config.polyak).add_(p, alpha=1 - config.polyak)
    if not all(
        bool(torch.isfinite(p).all()) for model in (actor, q1, q2) for p in model.parameters()
    ):
        raise FloatingPointError("nonfinite fitted TD3 weights")
    saved = {
        "schema": "rosclaw.parent_residual_td3.checkpoint.v1",
        "context_hash": context_hash,
        "torch_version": str(torch.__version__),
        "config": asdict(config),
        "shapes": shape_commitment,
        "actor": copy.deepcopy(actor.state_dict()),
        "target_actor": copy.deepcopy(target_actor.state_dict()),
        "q1": copy.deepcopy(q1.state_dict()),
        "q2": copy.deepcopy(q2.state_dict()),
        "target_q1": copy.deepcopy(tq1.state_dict()),
        "target_q2": copy.deepcopy(tq2.state_dict()),
        "actor_optimizer": copy.deepcopy(actor_optimizer.state_dict()),
        "critic_optimizer": copy.deepcopy(critic_optimizer.state_dict()),
        "generator_state": generator.get_state(),
        "total_actor_updates": total_actor,
        "total_critic_updates": total_critic,
    }
    return {
        "actor_weight": actor.weight.detach().numpy().copy(),
        "actor_bias": actor.bias.detach().numpy().copy(),
        "checkpoint": saved,
        "report": {
            "schema": "rosclaw.parent_residual_td3.numeric_report.v1",
            "context_hash": context_hash,
            "rows": len(s),
            "critic_context_dim": s.shape[1],
            "actor_feature_dim": f.shape[1],
            "action_dim": a.shape[1],
            "raw_action_semantics": "NORMALIZED_RESIDUAL_BEFORE_ENVIRONMENT_GUARD",
            "actual_critic_optimizer_updates": config.critic_steps,
            "actual_actor_optimizer_updates": accepted,
            "actor_attempts": attempts,
            "actor_trust_rejections": rejected,
            "total_actor_optimizer_updates": total_actor,
            "total_critic_optimizer_updates": total_critic,
            "critic_loss_first": critic_loss_first,
            "critic_loss_last": critic_loss_last,
            "replay_bank_actor_rms": rms(),
            "frozen_features_unchanged": True,
            "physical_provenance_verified": False,
            "physical_task_gain_proven": False,
            "hardware_authorized": False,
            "activation_ceiling": "SIM_ONLY",
        },
    }
