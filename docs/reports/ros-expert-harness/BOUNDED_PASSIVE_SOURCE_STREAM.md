# Bounded passive source-stream local acceptance

Status: **local passive source foundation only**. This does not close G3 or the
full live-supervision stage. Current ROS L1 monitoring, daemon-generation
binding, governed sink integration and L2 remain unimplemented/NOT_RUN. No
execution authority is created; REAL/hardware are unverified.

Frozen physical source: `145e9ed2b8449a8324ac3ab30292cec0f7bcf7fc`.
Its parent `fc7f2a23cea966cc87b9844fa15e1bf4339f8134` has identical Core/test
blobs; the only later change waits for actual ECM model removal/replacement
inside the disposable fault actor. All eight final Worlds used source 145e9ed2.
C++ source blobs exactly match tested SDK build source `2a95a1ea`; both separate
libraries built under `-Wall -Wextra -Werror`. The immutable image is
`sha256:d16320799584a60035548fb298243bdd8b0d2cb5760675a9cb46b608ea211e56`.

## Implemented boundary

The existing 4ec1afb1 passive guard now also pins full member ancestry and
requires consecutive sample cursors and bounded inter-sample wall time. An
SDK-only reader pins a separately accepted stopped G1/L1 segment-zero prefix,
its original SHA256, observer PID/independent random epoch and stream budget.
It checks exclusive file identity, append-only prefixes, SHA256 segment links,
current monotonic freshness at most 500 ms, cursor gaps, incomplete/invalid
lines and capacity. Failure latches permanently. Root motion may change; fixed
local geometry/sensor inventories/ancestry and static World poses may not.

SourceWitness keeps its original single-file mode by default. Explicit live
fixtures use 200 samples/segment, 15 MB/segment, at most eight ECM segments
(120 MB). Admission after rollover is deliberately refused in this first reader.
Contacts preserve original messages in at most four exclusive segments per
observer, each retaining the existing 30-MB/20,000-row/500-KB-line limits.
A durable hash-chain seals the contact segments. Capacity exhaustion fails
closed; `capacity_covers` is evidence budgeting, never motion authorization.

Generic source-only mode remains passive. There is no new Agent tool, velocity
publisher, execution registration, Runtime, permit or lease enable path in the
reader. Independent SDK fixtures explicitly own isolated Gazebo fault actors.

## Actual final Worlds and independent recomputation

Each World had a fresh UUID/run/Body. Initial G1 MATCHED and stopped L1 READY
were independently recomputed from original Native/ECM/physics files at the
original review time. The normal World remained continuously fresh for 75 s,
with 885 total ECM samples over five segments. All final contact messages were
reclassified against exact immutable horizontal-ground-plane collision names.
Every final case had zero unexpected contacts, zero dispatched goals/velocity
publications, disabled leases and stopped owned containers. These are passive
SIM results on the known fixture, not J100 motion or a contact-sensitivity test.

| Case | ECM samples | Original contact messages | Live result | Actual reason |
|---|---:|---:|---|---|
| positive | 885 | 27211 | CONTINUOUS | fresh continuous source |
| reset | 131 | 3304 | TERMINAL_LATCH | source stopped, stale or future |
| observer_kill | 162 | 4473 | TERMINAL_LATCH | source stopped, stale or future |
| stale | 162 | 4083 | TERMINAL_LATCH | source stopped, stale or future |
| entity_replaced | 152 | 3609 | TERMINAL_LATCH | unique dynamic robot required |
| sensor_drift | 148 | 3423 | TERMINAL_LATCH | unresolved loaded sensor type |
| clock_freeze | 166 | 3503 | TERMINAL_LATCH | incomplete, paused or reset source |
| budget_exhaustion | 200 | 5310 | TERMINAL_LATCH | source evidence capacity exhausted; recapture required |

Reset caused source loss/staleness in the original World; this is fail-close
evidence, not an isolated proof that a live reset-epoch sample was received.
Observer kill/stale also affect the in-process Gazebo World. Replacement
locks during removal; the independent actor additionally waited for the same
robot name with a different actual ECM entity ID. No automatic rearm occurred.
Sensor drift removes a real owned lidar-kind component via a separate,
explicitly loaded disposable plugin, leaving the observer read-only.

Independent review:
`recovery_2026-10-13/G3-live-continuity-eight-original-Worlds-independent-raw-review.json`.
It rechecks initial Native/L1, every ECM sample/segment/header/hash link, live
freshness timestamps, terminal causes, actual replacement, original contacts,
contact hash chains, source-only/lease traces and cleanup. It is RECOMPUTATION
of eight actual original Worlds, not eight additional Worlds.

## Preserved failures and repairs

Two initial positive Worlds failed when the independent contact logger reached
its existing 30-MB capacity and exited, taking the owned stack down. These
failures exposed a second evidence lifetime problem and motivated bounded
contact segments; no freshness or collision threshold was relaxed. Two early
negative runs failed before injection because initial physics evidence became
stale or was not time-corresponding to the selected ECM prefix. The supervisor
now captures a fresh corresponding pair before performing the complete L1
review once. A later driver-only replacement pass failed independent review
because creation acknowledgement did not prove an actual new entity. The
actor now independently waits for original ECM disappearance/new identity, and
the entire final seven-negative matrix was rerun on the corrected source.
All intermediate verifier/sealer failures and raw artifacts remain available.

## Software checks and archive integrity

- Exact-source focused contracts: 85 passed (26 continuity, 25 stream, 9 contact
  capture and 25 prior generic-source contracts).
- Core/Body/MCP/daemon regression: 1315 passed, 4 skipped, 11 deselected on fc7f2a23;
  source 145e9ed2 has byte-identical Core/test and checked execution code, apart
  from the separately revalidated SDK replacement fault actor.
- Skips: three optional e_urdf_zoo-package cases; one manifest without agent
  health checks. Eleven cases excluded by the recorded integration/acceptance/
  slow marker policy; actual isolated SIM Worlds are recorded separately.
- Compileall, changed Python Ruff/format and 121-file required Core mypy passed.
- Whole-repository Ruff remains the identical 290 baseline diagnostics; the
  same 606 baseline files require formatting. This is not whole-repository green.

Public original files are gzipped with stored/original SHA256 and original
size checks, excluding private homes, credentials, ledgers, pycache and symlinks.

- Final eight Worlds: `evidence/2026-10-13/G3-live-source145E9ED2-eight-original-Worlds-passive-only`, 1289 files;
  manifest SHA256 `6193b306964ce20cd870d28f5ef724677ce7c374410f657c4c9102ba30fd2b0a`.
- Intermediate attempts/failures: `evidence/2026-10-13/G3-live-source-original-intermediate-attempts-and-failures`,
  2014 files; manifest SHA256
  `80422a164b3eab9d79c8fc82db9253ba97d34f044df0eb903a4660d6dadb01ab`.

There were 20 actual new World starts across these final/intermediate attempts.
Only the final eight are accepted for the stated local passive-source scope.
Actual model calls: 0. Actual navigation actions: 0.

## Remaining exit criteria

G2 normal integration (#675) and its fresh actual-main known/J100 L1 still need
their complete original gates. This source-stream foundation requires review,
normal CI and actual-main proof. Full P0-1 additionally needs continuously
current ROS health and independently bound Body/run/daemon generation. G3 L2
then needs the explicit governed admission/sink, three J100 seeds with position
and yaw goals, and ten execution faults through canonical MCP/rosclawd.
G4 repair/rollback episodes and G5 actual paired model experiments remain NOT_RUN.
J100 prior L1 is adapted cross-family evidence; zero-code-change generalization
and REAL success are not claimed. `v1_done=false`.
