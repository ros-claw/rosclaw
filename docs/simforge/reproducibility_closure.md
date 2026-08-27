# SimForge reproducibility closure

`rosclaw.simforge.reproducibility` provides a task-neutral evidence boundary for
simulation and offline learning workloads. It was introduced after a continuous
multi-agent experiment showed that replaying twice inside one interpreter and
hashing only a few direct implementation files was not enough to identify the
actual numerical runtime.

The module does not know about a robot, sport, simulator score, or policy
architecture. A downstream application supplies those semantics. Core binds and
checks the parts that are common to every reproducible experiment:

- complete selected source trees, committed by relative file name and bytes;
- model, policy, dataset-snapshot, and predecessor-evidence artifacts;
- installed numerical dependency versions;
- Python implementation/version, platform, architecture, libc, CPU count, hash
  randomization, and numerical thread environment;
- the expected number of fresh worker processes;
- exact task-owned fields such as an evaluation object and trajectory digest;
- a hard `SIM_ONLY` ceiling with no hardware authorization.

## Minimal integration

The parent captures one immutable closure before launching workers:

```python
from pathlib import Path

from rosclaw.simforge import build_reproducibility_closure

closure = build_reproducibility_closure(
    source_trees={
        "application": Path("src/my_application"),
        "rosclaw_core": Path("vendor/rosclaw/src/rosclaw"),
    },
    dependency_packages=("mujoco", "numpy", "onnxruntime"),
    artifacts={
        "actor": Path("artifacts/actor.onnx"),
        "predecessor": Path("evidence/predecessor.json"),
    },
    expected_replays=3,
)
```

Each worker records its PID, `closure.closure_hash`, captured process contract,
task evaluation, trajectory digest, result, and authority fields. The parent or
an independent validator then derives the generic gates:

```python
from rosclaw.simforge import evaluate_cross_process_replays

verdict = evaluate_cross_process_replays(
    closure,
    worker_reports,
    exact_fields=("evaluation", "trajectory_digest"),
    launcher_process_id=parent_pid,
)
verdict.require_passed()
```

`require_passed()` rejects the run if the worker count is wrong, a PID is reused,
a worker is actually the launcher, the process contract or closure differs, an
exact field differs, any task result fails, or any worker claims hardware
authority. The verdict is itself canonical-JSON hashable.

## What this does not prove

- A content hash is not a signature or an independent trust anchor.
- Equal outputs do not prove that a task's physics gates are scientifically
  sufficient; the downstream validator must recompute those gates from raw
  evidence.
- Byte-exact replay is intentionally strict. A workload that is only
  tolerance-deterministic should expose a task-owned normalized comparison field
  rather than weaken Core's equality gate.
- `SIM_ONLY` evidence cannot authorize ROS, DDS, a vendor SDK, motors, or real
  hardware.

This separation lets football, manipulation, locomotion, navigation, and other
SimForge applications share one evidence closure without moving task-specific
scoring into ROSClaw Core.
