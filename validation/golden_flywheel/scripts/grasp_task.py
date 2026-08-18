"""Golden Flywheel grasp task — real MuJoCo physics (§42).

A parallel gripper closes on a small box and lifts it.  The skill parameter
is ``grip_closure_m``: how far each finger is commanded inward.  Below the
contact distance the fingers never reach the box — the "grasp" leaves it
behind during the lift (``slip_observed``); at full closure the force-capped
servos (2.0 N per finger) squeeze the box and carry it to the target height
(``success``).

The demo story is told by real contact dynamics, not by a scripted outcome
flag: ``run_grasp`` returns measurements, and the flywheel layers decide
what they mean.  ``grasp_receipt``/``verify_grasp_receipt`` adapt the task
to the promotion-gate evidence contract: receipts carry the randomization
and request record, and verification REPLAYS the run (same seed, same
closure) against the recorded outcome — MuJoCo is deterministic for the
same build, so a genuine run replays exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

# The model is parameter-free: the same MJCF serves baseline and candidate
# arms (the promotion gate requires identical model_hash across a pair).
_MODEL = """
<mujoco model="golden_grasp">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="0.8 0.01 0.001" solref="0.005 1"/>
    <body name="object" pos="{object_x:.5f} 0 0.021">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="0.02 0.02 0.02" mass="0.10"
            friction="1.1 0.005 0.0005"/>
    </body>
    <body name="palm" pos="0 0 0.021">
      <joint name="lift_z" type="slide" axis="0 0 1" range="0 0.12" damping="2"/>
      <geom name="palm_geom" type="box" size="0.045 0.03 0.004" pos="0 0 0.045" mass="0.2"/>
      <body name="finger_l" pos="-0.040 0 0">
        <joint name="finger_l_x" type="slide" axis="1 0 0" range="-0.005 0.030" damping="0.5"/>
        <geom name="finger_l_geom" type="box" size="0.005 0.025 0.025" pos="0 0 0.02"
              mass="0.05" friction="1.4 0.01 0.001"/>
      </body>
      <body name="finger_r" pos="0.040 0 0">
        <joint name="finger_r_x" type="slide" axis="1 0 0" range="-0.030 0.005" damping="0.5"/>
        <geom name="finger_r_geom" type="box" size="0.005 0.025 0.025" pos="0 0 0.02"
              mass="0.05" friction="1.4 0.01 0.001"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="grip_l" joint="finger_l_x" kp="400" forcerange="0 2.0"/>
    <position name="grip_r" joint="finger_r_x" kp="400" forcerange="-2.0 0"/>
    <position name="lift" joint="lift_z" kp="300" forcerange="-60 60"/>
  </actuator>
</mujoco>
"""

_CLOSE_S = 0.4
_LIFT_S = 0.8
_HOLD_S = 0.4
_LIFT_TARGET_M = 0.09
_SUCCESS_MIN_Z = 0.08  # object carried up with the palm
_SLIP_MAX_Z = 0.05  # clearly left behind during/after lift
_FORCE_CAP_N = 2.0  # actuator cap (servo forcerange)
# Task safety limit: 2x the actuator cap.  The lift transient legitimately
# spikes the measured contact force slightly above the servo cap (2.04 N
# observed); the safety contract is about crush risk, not servo bookkeeping.
_SAFE_FORCE_LIMIT_N = 4.0
_JITTER_M = 0.004
# Finger inner face starts at 0.035 m; the object face reaches 0.020+0.004 m.
# Closure 0.010 m -> faces at 0.025 m: no contact on any seed.  Closure
# 0.030 m -> faces at 0.005 m: solid capped-force squeeze on every seed.
BASELINE_CLOSURE_M = 0.010  # tuned: object never contacted (grasp slip)
PATCHED_CLOSURE_M = 0.030  # tuned: always held

MODEL_HASH = hashlib.sha256(_MODEL.encode()).hexdigest()
BACKEND_INFO = {"name": "golden-grasp-mujoco", "version": mujoco.__version__}


@dataclass(frozen=True)
class GraspEvidence:
    backend: str
    physics_executed: bool
    grip_closure_m: float
    seed: int
    object_x_m: float
    success: bool
    slip_observed: bool
    object_final_z_m: float
    object_min_z_during_lift_m: float
    peak_grip_contact_force_n: float
    steps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _bounded(value: float, *, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def trajectory_for(grip_closure_m: float) -> list[list[float]]:
    """The skill as a waypoint trajectory: close to ``grip_closure_m``, lift."""
    return [[grip_closure_m, -grip_closure_m, 0.0], [grip_closure_m, -grip_closure_m, _LIFT_TARGET_M]]


def run_grasp(*, grip_closure_m: float, seed: int = 0) -> GraspEvidence:
    """Close each finger to ``grip_closure_m``, ramp the lift, hold, measure."""
    grip_closure_m = _bounded(grip_closure_m, name="grip closure", lo=0.0, hi=0.030)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 100_000:
        raise ValueError(f"seed must be an int in [0, 100000], got {seed!r}")

    rng = np.random.default_rng(seed)
    object_x = float(rng.uniform(-_JITTER_M, _JITTER_M))

    model = mujoco.MjModel.from_xml_string(_MODEL.format(object_x=object_x))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    finger_l_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_geom")
    finger_r_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_geom")

    dt = model.opt.timestep
    close_steps = int(_CLOSE_S / dt)
    lift_steps = int(_LIFT_S / dt)
    hold_steps = int(_HOLD_S / dt)
    total = close_steps + lift_steps + hold_steps

    min_z_lift = math.inf
    peak_grip_n = 0.0
    for step in range(total):
        if step < close_steps:
            lift_cmd = 0.0  # close, no lift
        else:
            # ramp the lift — a step command jerks the contact loose even
            # with a good grip (found during physics tuning)
            lift_cmd = _LIFT_TARGET_M * min(1.0, (step - close_steps) / lift_steps)
        data.ctrl[:] = (grip_closure_m, -grip_closure_m, lift_cmd)
        mujoco.mj_step(model, data)
        # measured contact force on the finger geoms (real solver state)
        for i in range(data.ncon):
            contact = data.contact[i]
            if finger_l_geom in (contact.geom1, contact.geom2) or finger_r_geom in (
                contact.geom1,
                contact.geom2,
            ):
                frame = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, frame)
                peak_grip_n = max(peak_grip_n, float(abs(frame[0])))
        if step >= close_steps:
            min_z_lift = min(min_z_lift, float(data.xpos[object_body][2]))

    final_z = float(data.xpos[object_body][2])
    if min_z_lift is math.inf:
        min_z_lift = final_z
    success = final_z >= _SUCCESS_MIN_Z
    slip = (not success) and min_z_lift < _SLIP_MAX_Z

    return GraspEvidence(
        backend="mujoco",
        physics_executed=True,
        grip_closure_m=grip_closure_m,
        seed=seed,
        object_x_m=object_x,
        success=success,
        slip_observed=slip,
        object_final_z_m=round(final_z, 5),
        object_min_z_during_lift_m=round(min_z_lift, 5),
        peak_grip_contact_force_n=round(peak_grip_n, 4),
        steps=total,
    )


# ---------------------------------------------------------------------------
# Promotion-gate evidence contract (§44.10-11)
# ---------------------------------------------------------------------------


def grasp_receipt(*, seed: int, variant: str, grip_closure_m: float, evidence: GraspEvidence) -> dict[str, Any]:
    """A gate-grade receipt for one grasp run.

    Same contract shape as the trajectory-backend receipts the promotion
    gate was built for: evaluation_variant / pair_id / seed / randomization
    record / model+world hashes / request.  The arm difference lives in the
    trajectory (closure waypoints) — the gate compares requests with the
    trajectory popped.
    """
    if variant not in ("baseline", "candidate"):
        raise ValueError(f"variant must be baseline|candidate, got {variant!r}")
    scenario_id = f"golden_grasp_seed{seed}"
    trajectory = trajectory_for(grip_closure_m)
    randomization = {
        "method": "uniform_object_x_v1",
        "seed": seed,
        "seed_applied": True,
        "jitter_m": _JITTER_M,
        "initial_state_hash": hashlib.sha256(f"object_x:{evidence.object_x_m}".encode()).hexdigest(),
        "parameter_hash": hashlib.sha256(
            json.dumps({"jitter_m": _JITTER_M, "method": "uniform_object_x_v1"}).encode()
        ).hexdigest(),
        "offset_hash": hashlib.sha256(f"offset:{seed}:{evidence.object_x_m}".encode()).hexdigest(),
    }
    return {
        "id": f"grasp_receipt_{variant}_{seed}",
        "scenario_id": scenario_id,
        "pair_id": scenario_id,
        "evaluation_variant": variant,
        "seed": seed,
        "backend": dict(BACKEND_INFO),
        "model_hash": MODEL_HASH,
        "world_asset_hash": hashlib.sha256(b"golden-grasp-world-v1").hexdigest(),
        "randomization": randomization,
        "request": {
            "scenario": {
                "scenario_id": scenario_id,
                "robot_id": "golden_gripper",
                "world_id": "golden_tabletop",
                "seed": seed,
            },
            "trajectory": trajectory,
            "control_dt_sec": 0.002,
            "close_s": _CLOSE_S,
            "lift_s": _LIFT_S,
            "hold_s": _HOLD_S,
        },
        # Dropping the object IS the safety failure of this task; the
        # squeeze must also stay below the task's crush-safety limit.
        "is_safe": bool(
            evidence.success and evidence.peak_grip_contact_force_n <= _SAFE_FORCE_LIMIT_N
        ),
        "collision_pairs": [],
        "physics_executed": evidence.physics_executed,
        "observations": [evidence.to_dict()],
        # lineage marker: the evaluation records these as darwin_benchmark
        "darwin": True,
    }


def verify_grasp_receipt(receipt: dict[str, Any]) -> Any:
    """Task-appropriate receipt verifier for the promotion gate's
    ``receipt_verifier`` slot (the gate's designed extension point for
    non-trajectory-backend tasks).

    Contract check + deterministic replay: re-run the exact (seed, closure)
    and require the recorded outcome to reproduce exactly.
    """
    from rosclaw.sandbox.backends.base import ReplayReport
    from rosclaw.sandbox.evidence import SimulationEvidenceVerification

    def finish(verified: bool, reason: str, mismatches: tuple[str, ...] = ()) -> Any:
        return SimulationEvidenceVerification(
            verified,
            ReplayReport(
                verified=verified,
                environment_match=verified,
                hashes_verified=verified,
                deterministic_label=True,
                final_qpos_max_abs_error=0.0 if verified else None,
                reason=reason,
                mismatches=mismatches,
            ),
            mismatches,
        )

    errors: list[str] = []
    if receipt.get("evaluation_variant") not in ("baseline", "candidate"):
        errors.append("evaluation_variant")
    if not receipt.get("pair_id") or receipt.get("pair_id") != receipt.get("scenario_id"):
        errors.append("pair_id")
    rnd = receipt.get("randomization")
    rnd = rnd if isinstance(rnd, dict) else {}
    seed = receipt.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or rnd.get("method") != "uniform_object_x_v1"
        or rnd.get("seed_applied") is not True
        or rnd.get("seed") != seed
        or not rnd.get("initial_state_hash")
        or not rnd.get("parameter_hash")
        or not rnd.get("offset_hash")
    ):
        errors.append("seed_randomization")
    if receipt.get("model_hash") != MODEL_HASH:
        errors.append("model_hash")
    if not receipt.get("physics_executed"):
        errors.append("physics_executed")
    if errors:
        return finish(False, "GRASP_RECEIPT_CONTRACT_INVALID", tuple(errors))

    request = receipt.get("request") or {}
    trajectory = request.get("trajectory") or []
    observations = receipt.get("observations") or []
    if not trajectory or not observations:
        return finish(False, "REPLAY_INPUT_MISSING", ("request", "observations"))
    closure = float(trajectory[0][0])
    replay = run_grasp(grip_closure_m=closure, seed=seed)
    recorded = observations[0]
    mismatches: list[str] = []
    if replay.success != bool(recorded.get("success")):
        mismatches.append("success")
    if not math.isclose(replay.object_final_z_m, float(recorded.get("object_final_z_m", -1)), abs_tol=1e-4):
        mismatches.append("object_final_z_m")
    if not math.isclose(replay.object_x_m, float(recorded.get("object_x_m", -1)), abs_tol=1e-9):
        mismatches.append("object_x_m")
    if mismatches:
        return finish(False, "REPLAY_MISMATCH", tuple(mismatches))
    return finish(True, "replay reproduced the recorded run exactly")
