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

## Explicit offline ROS files fail closed

Inspection of the installed ROSClaw ROS connector found that a missing explicit
`ros compile --graph PATH` or `--manifest PATH` silently selected live rosbridge
discovery. This violated the declared offline mode and could turn a filename typo
into a network wait. Invalid manifest JSON could also escape as a traceback, and
an empty object could be accepted as an empty graph or unknown-body manifest.

The fix keeps explicitly selected files authoritative: missing or malformed files
return structured `ok:false` and exit code 1, without constructing a transport.
Graph files require ROS1/ROS2 version plus object lists for topics/services/actions;
manifest files require a named body and an object list of capabilities. Leaving
both offline flags unspecified preserves the explicitly live discovery path.

The focused tests initially recorded **17 failures / 4 passes** against the old
implementation. After the fix, **44 focused tests passed**, including all four
actual entrypoint/parser paths and a guarded test preserving the no-file live
branch. No ROS node, DDS transport, physical driver, or hardware command was run.

## Native offline interface chains

Twelve additional tasks (`CI01`–`CI12`) require calls to installed ROSClaw
interfaces rather than returning a formula answer. Six saved ROS graph cases
exercise graph compilation, persisted manifest loading, listing, capability
inspection, and rejection of deliberately unsafe offline velocity proposals.
They cover guarded commands, camera/laser artifacts, destructive services,
navigation action contracts, namespace collisions, and ROS1 type preservation.
Two negative cases require structured CLI rejection for invalid graphs and
missing manifests. Four Practice cases require actual recording and strict
verification of a valid fixture, ingestion rejection for a missing envelope,
strict detection of deliberate raw-event corruption, and preservation of a
failed trial's evidence without claiming physical success.

The external grader binds the unchanged supplied input to the generated manifest,
compilation JSON, listed capabilities and per-capability inspection JSON. It
checks semantic risk/guard/artifact projections and actual native-session bash
tool calls. A correct-looking answer or artifact without those interface calls
is insufficient. Practice grading checks actual durable event records and strict
verification/exit-code evidence. Unit-test transcripts are explicitly labelled
as unit tests and are never counted as real model trials.

These are real ROSClaw **offline interface integrations**, still backed by
synthetic data and labelled FIXTURE_ONLY. A saved ROS1/ROS2 graph is not transport
discovery. Installed Jazzy schemas are present on this host, but these tasks do
not open DDS, instantiate ROS nodes, or infer live support. Direct multimodal
VLA/VLN inference and physical navigation remain outside the evidence scope.
