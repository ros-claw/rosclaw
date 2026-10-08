# N01: passive coverage causal audit

Status: fresh SIM physical and diagnostic acceptance PASS on frozen source
`2dcb77bd03de034963776c7e2e89db5070d3c57e`. Expanded local regression passes; latest-head CI is pending and the PR merge
gate remains open.

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

## Recorded physical observation

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

## Formal accepted run

Run `n01-waffle-004`, ID `f1e279292c684a769b569d84d9381c6a`, uses a clean
source freeze and unchanged rebuilt image
`sha256:c31355f34739eb4ea8b60de1414c57ce0eea8225aa9e3ce854b7ceb66ca4376e`.
The [public evidence archive](runs/n01-waffle-004/README.md) includes the original
canonical artifact, receipts, verifier result, append-only streams and source
hashes. Authentication, action tickets and private ledger/model data are excluded.

| Measured quantity | Result |
| --- | ---: |
| Fixed denominator | 3,576 cells |
| Main observed coverage | 2,047 cells / 57.2427293065% |
| Main distance / SIM span | 10.5665212995 m / 65.6 s |
| Final observed coverage | 3,505 cells / 98.0145413870% |
| Total observed distance / SIM span | 57.9029962809 m / 554.84 s |
| Canonical mission duration | 560.255835 s |
| Independent pose samples | 11,098 |
| Contact count / trace gaps | 0 / 0 |
| Captured feedback / repair goals | 54,747 / 26 |
| Audit loss / source binding errors | 0 / 0 |
| Canonical verifier independent replay | Exactly equal |
| Receipts / Practice / Memory reopen | 3 COMPLETED SIM / SUCCESS / 5 queries |
| Independent post-cleanup 3 s displacement / yaw | 0 m / 0 rad |

Canonical duration and observed SIM span measure different intervals; neither is
substituted for the other. Segment intervals include waiting and partition all
observed distance, rotation and SIM duration exactly once.

Of the 1,529 first-pass missed cells, 1,425 (93.20%) lie outside the recorded
ideal plan sweep; 104 (6.80%) remain UNKNOWN. The recorded plan predicts 2,141
cells (59.8713646532%) and is 9.6929116639 m long. The upstream server reserves
0.5 m headland for turning and generates interior swaths without an explicit
headland cleaning pass. This explains the predominant plan omission. It does
not establish that shrinking the headland is safe, nor assign the remaining
104 cells to tracking error. The plan-to-execution
[overlay](runs/n01-waffle-004/plan_execution_overlay.svg) displays both masks
and trajectories; predictions never enter canonical coverage credit.

`n01-waffle-001` failed at startup; no cleaning was dispatched. Pilot 002
physically passed but lacked a clean source freeze and closing plan summary.
Run 003 physically completed but its closing plan summary was missing.
Both have incomplete audits, retained separately. Run 004 fixes the shutdown
flush path and supplies complete diagnostic evidence. No missing historical
heading or internal swath/controller stage was invented.

Physical acceptance belongs to the recorded frozen commit, not a subsequently
edited HEAD. Later runtime changes hash the audit filename to prevent path
traversal; execution/repair/goal/service functions, witness, stack, daemon and
logger remain identical. Later offline processing repairs saved-verification
lookup and source binding; it does not change the frozen physical episode.

## Validation and next gate

ROS connector regression: 303 passed / 11 deselected. Audit tests: 6 passed,
including immutable snapshots, hash tampering, unfinished EOF, bounded loss,
continuous prediction, prediction isolation, failed flush preserving canonical
result and lock release, and malicious action-ID filenames. Targeted source
type checks and changed-file Ruff/format/diff checks passed. Counts overlap and
are not summed. Formal public JSON schemas and segment-total invariants are
also checked against the complete run.

Expanded architecture/security/agentd regression: **1,025 passed, 8 skipped,
2 deselected**, 5 warnings, 3,095.80 s. This run overlapped N01 development;
frozen latest-head CI remains the authoritative merge gate. Warning categories
are a deprecated class fixture, jieba/pkg_resources deprecation and async
subprocess finalizers after event-loop close. No test failed.

Latest-head CI is still running; it must pass before PR #621 is merged. The
overlay was rendered and visually inspected in Chromium; labels and curves
are readable and the large omitted headland is visible.

N02 has 15 offline candidates generated by the actual pinned Fields2Cover
installation, varying headland and angle. Several high-coverage candidates
have insufficient or negative sampled body clearance and are rejected. A
45-degree swath angle with the original 0.5 m headland predicts 77.49%, but
needs fresh controlled physical trials. Offline geometry is not physical
acceptance; no efficiency improvement is claimed.

Efficiency, dynamic occupancy, unknown Body, Memory causal benefit and A/B
remain unaccepted; `v1_done=false`.
