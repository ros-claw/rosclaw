# G1 GoalForge Phase 4

GoalForge is ROSClaw’s simulation-only G1 penalty-kick verification loop. It
keeps the public RoboNaldo whole-body kick policy fixed and permits adaptation
only through bounded, interpretable stance, heading, contact, swing, and
recovery parameters.

## Trust boundary

- No real robot transport is opened. All permits use the `SHADOW` evidence
  domain and explicitly deny hardware authorization.
- Final task truth comes from the qualified 29-DoF G1 MuJoCo model at 500 Hz,
  not from CUDA proxy screening or visualization.
- Unitree DDS verification is restricted to `lo`, a nonzero DDS domain, and
  the canonical `rt/lowcmd` / `rt/lowstate` topics.
- Candidate generation sees noisy ball/target observations and Twin beliefs,
  but never hidden mass, friction, latency, Holdout seeds, or per-case Holdout
  outcomes.
- Raw trajectories, private Holdout rows, signing keys, and receipts must be
  written outside the source checkout.

## Implemented loop

The flagship loop performs:

1. a physically verified fixed-prior target miss;
2. `FailureSignatureV3` routing with a retry budget of at most two;
3. a residual-only `KickTwinBelief` update;
4. bounded How and Auto candidate generation;
5. at most two audited Sandbox candidate attempts and a same-seed physical
   retry;
6. a first-shot evaluation at a new ball/target location;
7. independent trajectory verification and strict replay receipts.

Practice records store control-frequency joint, torque, IMU/pelvis, COM, foot
contact, ground reaction, slip, ball, contact-point, impulse, and action arrays
in compressed NPZ. Low-frequency causal events refer to those arrays by hash.
The grouped split keeps scenario IDs disjoint across Development, Validation,
and the mode-0600 Private Holdout. The acceptance flywheel uses twelve
distinct scenarios: eight distinct teacher contexts, two Validation scenarios,
and two undisclosed Holdout scenarios. Baseline/candidate records from the
same scenario do not count as separate teacher contexts.

The learned `G1ShotAdapter` is a one-hidden-layer MLP. Every learned output
binds the exact Dataset Snapshot hash, is projected back into the adapter’s
safe parameter bounds, reports uncertainty, and can only be activated for the
qualified Body and fixed-prior hashes after zero-fall/zero-torque validation.
After activation, the flywheel resolves the Champion through the Registry and
runs an ordinary validation task; `active=true` alone is not acceptance.

The 100-pair recovery validator retains every Sandbox attempt, including
rejected overshoot or joint-limit candidates. A conservative second candidate
may be evaluated only inside the original FailureSignature retry budget.
Promotion consumes the final safe candidate and the complete attempt audit;
unsafe intermediate candidates are not relabeled as executed robot actions.

## Reproduction

Set the public external checkouts explicitly when they are not under
`/code/rosclaw/phase4_references`.

```bash
export ROSCLAW_G1_ASSET_ROOT=/path/to/RoboNaldo_Deploy
export ROSCLAW_SIMFORGE_REFERENCE_ROOT=/path/to/phase4_references

.venv/bin/python -m rosclaw.entrypoint simforge doctor g1-goalforge --all \
  --output /tmp/goalforge-doctor.json

.venv/bin/python -m rosclaw.entrypoint simforge validate g1-goalforge \
  --pairs 100 \
  --output /tmp/goalforge-recovery-100.json

.venv/bin/python -m rosclaw.entrypoint demo run g1-goalforge \
  --target-zone random \
  --failure-to-success \
  --live-dashboard \
  --output-dir /tmp/goalforge-demo

.venv/bin/python -m rosclaw.entrypoint practice dataset build \
  --task g1_penalty_kick \
  --generation 3 \
  --output-dir /tmp/goalforge-practice-g3

.venv/bin/python -m rosclaw.entrypoint evolution run \
  --task g1_penalty_kick \
  --generation 4 \
  --gpus 0,1,2,3 \
  --output-dir /tmp/goalforge-evolution-g4

.venv/bin/python -m rosclaw.entrypoint chaos run g1-goalforge \
  --faults agent-kill,worker-crash,dds-loss,state-stale \
  --output-dir /tmp/goalforge-chaos

.venv/bin/python -m rosclaw.entrypoint proof build g1-goalforge \
  --demo /tmp/goalforge-demo/goalforge-demo.json \
  --recovery /tmp/goalforge-recovery-100.json \
  --flywheel /tmp/goalforge-practice-g3/goalforge-flywheel.json \
  --memory /tmp/goalforge-evolution-g4/memory-ablation-100.json \
  --four-gpu /tmp/goalforge-evolution-g4 \
  --agreement /tmp/goalforge-evolution-g4/cpu-gpu-label-agreement.json \
  --continual /tmp/goalforge-evolution-g4/continual-g0-g10.json \
  --chaos /tmp/goalforge-chaos/goalforge-chaos.json \
  --output-dir /tmp/goalforge-proofs

.venv/bin/python -m rosclaw.entrypoint proof replay /tmp/goalforge-proofs \
  --modules body,provider,failure_router,sandbox,practice,memory,know,how,auto,darwin,registry,rosclawd

.venv/bin/python -m rosclaw.entrypoint promotion evaluate g1-goalforge \
  --doctor /tmp/goalforge-doctor.json \
  --recovery /tmp/goalforge-recovery-100.json \
  --flywheel /tmp/goalforge-practice-g3/goalforge-flywheel.json \
  --four-gpu /tmp/goalforge-evolution-g4 \
  --continual /tmp/goalforge-evolution-g4/continual-g0-g10.json \
  --chaos /tmp/goalforge-chaos/goalforge-chaos.json \
  --proofs /tmp/goalforge-proofs \
  --output /tmp/goalforge-promotion.json

.venv/bin/python -m rosclaw.entrypoint evolution export /tmp/goalforge-demo \
  --format showcase \
  --output /tmp/goalforge-public-showcase

MUJOCO_GL=egl \
.venv/bin/python -m rosclaw.entrypoint evolution export /tmp/goalforge-demo \
  --format video \
  --output /tmp/goalforge-public-video.mp4
```

## Public-project provenance

The external assets are not vendored into ROSClaw. Qualification records their
Git commits and content hashes.

- Unitree `unitree_rl_mjlab`: G1 locomotion/motion-imitation and multi-GPU
  training contract.
- Unitree `unitree_mujoco`: official 29-DoF G1 model and SDK2 DDS bridge
  semantics.
- Unitree `unitree_sim_isaaclab`: optional G1 visual/DDS second-backend
  contract. Isaac Lab runtime absence is reported as optional, never silently
  relabeled as executed.
- OpenDriveLab RoboNaldo: public 29-joint free-kick policy, motion prior, and
  G1-with-ball scene used by the strict physics backend.

GPU shards are signed screening evidence only. A complete run requires all
four physical GPU UUIDs, at least 1,000 independent scenario commitments,
G0–G10 coverage, an undisclosed Private Holdout, and final CPU MuJoCo checks.
The first CPU sample is calibration-only. Acceptance uses a disjoint,
class-balanced CPU validation split and requires high agreement for both safety
and candidate-success labels. Calibration-before and calibration-after reports
must both be retained.

`proof replay` does not trust a stored `level: E5` string. It recomputes the
ProofBundle hash and independently re-derives each requested module level from
matched counterfactual, fault-injection, and replay fields. Missing modules,
tampered JSON, failed faults, or unmatched counterfactuals fail closed.
Promotion additionally requires the simulation-only Doctor report and checks
that its qualified Body and kick-prior hashes match both the activated
Champion and the independently replayed ProofBundle. G15 requires all twelve
GoalForge modules at E5.

The showcase export contains both the evidence manifest and a dependency-free
`index.html`. Its left panel draws the three recorded MuJoCo ball trajectories,
the center panel shows the causal module chain, and the right panel shows
Verifier/Receipt metrics.

The video export reconstructs the recorded pelvis, 29-joint, and ball poses in
the qualified MuJoCo scene. It emits H.264 MP4 with a contact slow-motion
window, target marker, ball trail, verifier-derived shot labels, and a
hash-bound JSON manifest. Both exports are downstream visualizations of
recorded evidence and never produce task labels or Promotion evidence.
