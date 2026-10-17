# RK-01: read-only ROS Operational Graph and missing-evidence plans

This is the first bounded 1018 work package, based on main `be5d36ff`. It implements RK-00/RK-01 only. Neither active probes, interventions, new Worlds nor model benchmarks execute in this phase. Prior v1/1016 findings remain frozen.

## Public Python API

```python
from rosclaw.connectors.ros.intelligence.operational_graph import (
    compile_operational_graph, summarize_operational_graph,
)
from rosclaw.connectors.ros.intelligence.missing_evidence import describe_missing_evidence

graph = compile_operational_graph(
    sealed_model, original_urdf_bytes, explicit_navigation_requirements,
    source_ref="original:sealed-model-reference",
    now=explicit_aware_original_review_clock,
    evidence_domain="SEALED_REPLAY",
)
projection = summarize_operational_graph(graph)  # <=6000 serialized characters
plan = describe_missing_evidence(graph, max_probes=8, budget_sec=30)
```

This is an offline projection of existing navigation requirements, **not** a general task/application compiler or a new ROS runtime. No ROS client dependency is imported. The input is revalidated against its original snapshot hash, and the existing Case compiler remains unchanged. All original six constraint verdicts (source plus five repair mechanisms) are preserved verbatim; additional findings are distinct and do not grant staging or motion permissions.

Typed entities cover exact observed node/topic/service/action/frame identities plus original URDF, source and, when supplied, historical run/Body/world references. Endpoint identities retain actual GIDs when present; duplicate node-only endpoints stay separate and ambiguous. Publisher→subscriber edges join only exact resolved topic names. The four-policy directional QoS check covers reliability, durability, liveliness kind and resolved deadline; full RMW compatibility and delivery remain UNKNOWN/NOT_MEASURED. No unrecorded lease duration, middleware default, plugin consumption or physical effect is inferred.

TF clock findings require a fresh snapshot/edge and observed aligned node time domains. Static transforms are not evaluated as dynamic-age samples. TF duplicate parent/authority witnesses can violate an invariant; absent per-message publisher attribution never proves uniqueness. Sensor receive age is separate from message/ROS age, and historical time does not become current time. The existing native probe lacks the full TF attribution window and complete QoS record; plans honestly mark `PROVIDER_EXTENSION_REQUIRED`.

`describe_missing_evidence` groups UNKNOWN findings into source review, endpoint QoS, TF-authority, clock/TF or native diagnostics requests. Requests carry exact known targets, findings, required evidence, source/snapshot/graph binding, at most five seconds each and at most 30 seconds total. Excess groups are explicitly deferred. Source mismatch blocks ordinary continuation and requests only source review. All requests are **PROPOSED_NOT_EXECUTED**, with side effects NONE and execution authority NONE. These are measurement specifications, not shell/service/action commands or claims of available/executed experiments.

Both the direct Python compiler and replay file reader reject snapshot/URDF/requirements/source-review inputs >5 MB. Bounds also reject excessive observation/endpoint counts, >4096 potential dataflow pairs and oversized identities. Full graph collection limits are explicit in the schema. Agent summaries report omitted finding counts and reference the complete graph; they cannot masquerade as complete evidence.

## Offline original-evidence replay

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmarks.ros_expert_ablation.replay_operational_graph \
  /path/to/SHA256-pinned-manifest.json /path/to/new-output-directory
```

Each manifest entry supplies SHA256-pinned observation, original URDF, requirements and an aware review clock. Optional source review is separately pinned. `sealed_system_model` verifies the original seal. `native_probe` uses the existing pure builder and writes a separately named normalized model, linked by a content reference to the original native SHA256 and derived snapshot hash. The original raw native path is never falsely used as a normalized-document locator. Original Case documents are exported too, making every graph evidence pointer independently resolvable. Inputs and pre-existing output directories are never overwritten.

The local manifest `recovery_2026-10-17/RK01-original66-replay-inputs-v1.json` contains:

|Original collection|Entries|Interpretation|
|---|---:|---|
|Frozen original G5 model snapshots|54|Offline regression of original observed constraints; no model calls|
|Original G3 native snapshots|2|Historical healthy-observation and sensor-loss controls, not current readiness|
|Original G4 five mechanisms, before and rollback|10|Historical source-training snapshots; not new interventions|

All 66 preserve the unchanged Case verdicts. Each evidence pointer is resolved against the sealed model or separately exported Case and its value hash is checked. Agent projections stay within 6000 characters. A separate process repeats the frozen source replay and must produce identical graph, plan, Case, normalized-model and summary bytes. Current source/main acceptance records and hashes live separately under `recovery_2026-10-17/`; development runs remain labeled development.

New native-shaped synthetic tests cover directional QoS faults, unresolved defaults, TF time mismatch, duplicate TF authority, missing identity, namespace/remap, wrong source, stale/future snapshots, hash tampering, cross-link corruption, bounds, deferred targets, pure no-I/O behavior and Core import without ROS. These are protocol fixtures, **not** actual injected SIM faults.

## Validation and limitations

The new focused suite has 36 tests. The final combined local ROS/benchmark/architecture suite, Practice checks, prescribed mypy, four new-module mypy, compileall, changed-file Ruff and normal CI are recorded with original logs. The architecture console absence is a pre-existing skip; Practice's optional skips and marker deselections are reported separately, never counted as PASS.

Repository-wide Ruff/format findings are compared byte-for-byte (format footer count excluded) against the pinned baseline. Existing 291 Ruff findings and 607 format candidates are not repaired as unrelated scope. `src tests` and all changed Python files must pass lint.

NOT_RUN: new native provider collection, active hypothesis/probe execution (RK-02), counterfactual interventions (RK-03), new Body/application compilation, cross-Body Memory promotion, physical hardware and new model benchmark. No change to production kernel, controller speed, proof250ms/fsync, rosclawd gates, World lifecycle, original five repair templates or experiment definitions. RK-01 establishes inspectable graph/measurement contracts; it does not prove a root-cause or model-effectiveness breakthrough.

## Official semantics

- [ROS 2 Jazzy QoS documentation source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst): directional requested/offered policy compatibility; all matching policies matter.
- [ROS clocks](https://design.ros2.org/articles/clock_and_time.html): ROS, System and Steady domains are distinct.
- [Jazzy TF and time](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Intermediate/Tf2/Learning-About-Tf2-And-Time-Cpp.rst): transforms have temporal constraints as well as topology.

Versioned schemas: `RosOperationalGraphV1.schema.json` and `MissingEvidencePlanV1.schema.json`, generated from strict frozen Pydantic contracts. Schema contents and a round trip are checked in source/main validation.
