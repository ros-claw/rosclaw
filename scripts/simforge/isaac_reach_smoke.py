#!/usr/bin/env python3
"""Finite, evidence-producing Isaac Lab UR10 reach qualification run."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Reach-UR10-Play")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    output = args_cli.output.expanduser().resolve()
    if output.is_relative_to(root):
        parser.error("--output must point outside the source checkout")
    if not 1 <= args_cli.num_envs <= 8192:
        parser.error("--num-envs must be in [1, 8192]")
    if not 1 <= args_cli.steps <= 100_000:
        parser.error("--steps must be in [1, 100000]")
    if args_cli.num_envs * args_cli.steps > 10_000_000:
        parser.error("--num-envs * --steps cannot exceed 10000000")
    torch.manual_seed(args_cli.seed)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    started = time.perf_counter()
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            observation, _ = env.reset(seed=args_cli.seed)
            finite = _tree_finite(observation)
            reward_sum = 0.0
            completed_episodes = 0
            step_started = time.perf_counter()
            for _ in range(args_cli.steps):
                with torch.inference_mode():
                    actions = (
                        2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                    )
                    observation, rewards, terminated, truncated, _info = env.step(actions)
                finite = (
                    finite and _tree_finite(observation) and bool(torch.isfinite(rewards).all())
                )
                reward_sum += float(rewards.mean().item())
                completed_episodes += int(torch.count_nonzero(terminated | truncated).item())
            step_wall = time.perf_counter() - step_started
            payload = {
                "schema_version": "rosclaw.simforge.isaac_reach.v1",
                "task": args_cli.task,
                "backend": type(env_cfg.sim.physics).__name__,
                "device": str(env.unwrapped.device),
                "num_envs": args_cli.num_envs,
                "steps": args_cli.steps,
                "env_steps": args_cli.num_envs * args_cli.steps,
                "wall_time_sec": time.perf_counter() - started,
                "step_wall_time_sec": step_wall,
                "env_steps_per_sec": args_cli.num_envs * args_cli.steps / max(step_wall, 1e-9),
                "mean_reward": reward_sum / args_cli.steps,
                "completed_episodes": completed_episodes,
                "finite_state": finite,
                "seed": args_cli.seed,
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["evidence_hash"] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        finally:
            env.close()
    if not math.isfinite(float(payload["mean_reward"])):
        payload["finite_state"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["finite_state"] else 2


def _tree_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tree_finite(item) for item in value)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
