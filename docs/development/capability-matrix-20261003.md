# Native-agent offline capability matrix, 2026-10-03

The matrix adds 60 independent robotics contract directions (`C01`–`C60`), plus
12 composite stress tasks (`CX01`–`CX12`), to the existing HarnessBench runner.
Native ROSClaw supplies every answer. The benchmark authors supply synthetic
inputs, public contracts, and private grading witnesses; they do not implement
robot controllers, perception pipelines, policies, or task answers for ROSClaw.

These are **FIXTURE_ONLY** agent reasoning and interface-contract probes.
Passing image byte/depth/point-cloud contracts does not prove camera operation
or visual perception. VLA cases inspect action ordering, units, and freshness;
they do not run a VLA network. VLN cases use structured landmark observations;
they do not send images to a vision model or execute navigation. Live ROS1,
ROS2 transport, hardware integration, training, and policy rollout are **NOT_RUN**
by this suite. The existing 32-task benchmark contains distinct simulation and
receipt-based tasks; its results must be reported separately from this layer.

Run with a configured real model, preserving the existing model-key discipline:

```sh
python scripts/harnessbench_run.py --legs B --model kimi-k3 \
  --tasks C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,C12 --runs 1 \
  --out /tmp/rosclaw-capability-matrix
```

`benchmarks/harnessbench/capability_cases.py` contains the 60 reviewed directions;
`capability_hard_cases.py` contains the composite probes. The external
`capability_matrix.public_manifest()` exports trigger inputs, contracts, and
pass/fail boundaries without reference answers. Do not stage the Python catalog
or grader in the evaluated workspace. The agent receives only `task.md` and
`input.json`; outputs require `answer.json` with a semantic result, exact-input
SHA256, truthful fixture scope, and an explanation. Inputs are immutable.

The oracle checks all result fields recursively, including numerical tolerances,
array dimensions, exact object keys, boolean-vs-numeric distinction, nonfinite
values, exact input bytes and hash, and evidence scope. It rejects claimed LIVE
or SIM evidence from fixture input. Grader tests retain positive witnesses and
mutations for missing fields, changed inputs, unsupported scopes, stale binding,
nonfinite values, type confusion, missing reasons, and malformed JSON.

The 12 composite tasks preserve the initial C01–C60 inputs and contracts, so
running the harder batch never changes the meaning of an earlier receipt.
They combine coordinate transforms and depth units; ROS namespaces, domains and
QoS; clock offsets, pairing and reuse; bag reset epochs, duplicate packets and
loss; actuator-local clipping, gear ratios and summed joint limits; sequential
Kalman prediction/update; endian and stride handling; VLA ordering, degrees,
freshness and replay; moving-obstacle stopping distance; wrench rotation and
translation; VLN room and temporal grounding; and joint-name/time/effort checks.

A configured model run is separate from local grader validation. Local unit tests
are not agent successes; missing credentials must remain NOT_RUN. Preserve failed
trials and record retries as new trials. A formula-probe PASS is useful component
evidence, but cannot establish the upper limit of embodied control or a framework's
whole-system reliability.
