# N01: passive coverage causal audit

Status: implementation and focused regression accepted; fresh physical run in
progress. This report does not yet declare the N01 live/merge gate complete.

## Implemented

- Bounded asynchronous hash-chained JSONL logging, exclusive creation, immutable
  producer snapshots, explicit dropped/writer-error summaries. Audit callbacks
  perform no disk IO; they cannot update the verifier or control motion.
- Nav2 main and repair feedback retained with goal IDs, observed clock and
  exact consumer sample offsets. Stage names describe an executor interval,
  not an independently verified TaskKernel stage success. Internal controller,
  headland/connector execution and recovery-state details remain UNKNOWN when
  the exposed action/debug messages do not establish them.
- Passive pinned-server debug path, field, planning field and swath collection;
  append-only versions retain yaw, header time, capture time, source and hash.
  No upstream planner patch, upgrade or new motion endpoint.
- Public daemon source/config freeze with git SHA, dirty-tree marker, run ID,
  geometry/map hashes; no authentication or private model data.
- Read-only auditor validates the canonical artifact hash and exactly replays
  CoverageVerifier. An ideal brush-on plan sweep is separately predicted, never
  counted as measured credit. Missing events/yaw do not acquire guessed values.
- Per-interval measured distance/rotation/time/coverage gain, threshold-labelled
  movement classification and mutually exclusive first-pass missed-cell labels.
  Planned-but-missed cells remain UNKNOWN without a causal observation. Audit
  completeness is separate from canonical task success.

## Preliminary physical observation

Instrumentation pilot `n01-waffle-002` uses the unchanged room, Body, attachment,
algorithm and rebuilt image. Its upstream field debug polygon spans [-1.5,1.5]
in x/y. The planning-field polygon spans [-1,1] after the configured 0.5 m
headland removal. Four swaths are observed at y approximately -0.775, -0.325,
0.125 and 0.575. The recorded 98-point coverage path is 9.692911663879869 m.
Its ideal measured-geometry brush-on prediction covers 59.87136465324385%;
the first actual recovery checkpoint covers 57.07494407158836%.

This establishes a large gap already in the recorded plan, separately from
execution error. It does not establish that all tracking loss has one cause,
nor that a smaller headland is safe/effective without a controlled follow-up.
Historic 57.27069351230425% is exactly reproduced from the archived trajectory;
the new pilot is a fresh episode with its own measured result, not the same
floating-point value retroactively attributed to old evidence.

The startup failure in `n01-waffle-001` is retained. The witness originally
imported an eager diagnosis initializer requiring absent Pydantic; making
diagnosis exports lazy fixed the pure passive import in the unchanged image.
No dependency installation or control guard relaxation was used.

## Validation

ROS connector regression before the final added projection test: 301 passed /
11 deselected. Focused audit/native regression: 64 passed. Current audit tests:
5 passed, including integrity mutation, unfinished EOF, invalid/oversize loss,
input mutation, continuous path projection and prediction isolation. Targeted
source type checks and changed-file Ruff/format/diff checks passed.

Architecture/security/agentd checks and fresh complete physical acceptance are
still running; their final results must be recorded before merging. The
existing full CI remains the release gate. These counts overlap and are not
summed. Efficiency, dynamic occupancy, unknown-Body, Memory causal benefit and
A/B remain unaccepted; `v1_done=false`.
