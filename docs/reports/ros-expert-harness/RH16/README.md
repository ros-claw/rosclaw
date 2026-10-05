# Extended acceptance after PR #617

Follow-up: the original task sentences `完成整个房间清扫。` and `clean the entire
room` still failed under the legacy tokenizer even after correcting the stored
instruction. [Root-query comparison](bilingual-root-query/acceptance.json) uses
the exact historical tokenizer from a54c83e3 and the same actual successful
episode. Both queries fail before and retrieve the original episode after
reusing the existing bilingual tokenizer and clean/cleaning aliases. Legacy
terms remain available; Chinese segmentation also works with the existing
bigram fallback when jieba is absent. Tests cover the normal retrieval path and
keyword-only/no-jieba installations. [Regression](bilingual-root-query/regression.log.gz):
202 passed; changed Memory/Practice type checking passed. The earlier results
below describe their recorded revisions; final PR checks must cover this fix too.

The implemented path has additional accepted gates; the complete research plan
remains **PARTIAL**. [Machine-readable summary](acceptance_summary.json) distinguishes
live simulation, historical replay, offline Body compilation and GPU activity.
[Artifact manifest](artifact_manifest.json) records original and archived SHA-256
hashes. Gzip archives preserve original bytes; control-ledger authentication keys,
model credentials and raw model transcripts are excluded.

## Accepted results

| Gate | Evidence and result | Scope |
| --- | --- | --- |
| Restart with the same authenticated ledger | [Restart002](restart-002/safety.json): SIGKILL during actual Gazebo motion, independent standstill, new daemon generation, ledger integrity verified, DISARMED, interrupted action FAILED / DAEMON_RESTART_INTERRUPTED | Live owned SIM, same-UID process boundary |
| No revival of old authority | Old heartbeat rejected with SESSION_NOT_FOUND; old action renewal rejected with ACTION_NOT_ACTIVE. Three seconds after restart: zero displacement/yaw change, cleaner disabled, no lease, zero contacts | Transport failures cannot satisfy the rejection gate |
| Memory retrieval and persistence | [Replay](memory-reuse/acceptance.json): three cleaning queries returned no baseline matches. After semantic projection, reopen and double replay, all retrieve the original accepted mission; one record remains, original verifier retained, receipt duration 675.39052 seconds | Actual Native011 historical verification, not another physical mission |
| KNOW/HOW feedback persistence | [Write](knowledge-feedback/write.json) and [separate-process read](knowledge-feedback/read.json): sourced stored pack, HOW submission, exact duplicate ignored, conflicting same ID rejected, one durable governance record | Actual episode references; unknown contribution, used_by_agent=false, manual review, no automatic knowledge mutation |
| Explicit CUDA activity | [Copy trace](cuda-copy/copy-trace-acceptance.json): three actual component PIDs, 301 Resize kernels, 301 input Device-to-Host copies (1,872,460,800 bytes), 301 output Host-to-Host copies (117,028,800 bytes), shareable-handle import/export calls | 300 validated frames plus one warmup, real GPU graph under Nsight |
| Second vendor Body contract | [Contract](second-body-contract/acceptance.json): installed ROBOTIS Burger and existing Waffle have distinct vendor URDF hashes, effective Body hashes, collision boxes and wheel separations (0.160 vs 0.288 m) | Offline compilation; explicit fixture radii and simulated cleaner declarations |

Memory's success record previously used the generic instruction `ROS ros.expert`
and duration zero. The ROS Practice adapter now identifies `coverage.execute`,
adds bilingual cleaning semantics, and obtains elapsed time from canonical receipt
timestamps. Missing, invalid or reversed timing remains zero and cannot invent a
duration. The raw verification is retained unchanged. Tests reopen a real SQLite
store, retrieve both English and Chinese queries, and check timing/error cases.
No existing historical database is silently migrated: acceptance replays into a
private SQLite backup only. Query timings describe this single-record store and
the existing result cache; they do not prove an agent or mission speedup.

Native011 is also [archived here](native-011-watchdog-full-pass/acceptance.json).
It passed at 98.07046979865772% independent coverage, zero contacts and complete
observations with the controller watchdog, dynamic obstacle, three completed
canonical actions, Memory success, Practice success and TaskKernel SUCCEEDED.
Together with Native004 and Native010, three complete actual-model missions have
accepted evidence, two with the watchdog. Native011 predates this Memory fix and
ran with knowledge disabled; advice contribution is explicitly unmeasured.

## Partial and retained failed gates

The GPU profiler reports that Unified Memory tracing is unsupported by the
current driver/configuration. CUDA activity and all three component processes are
captured, but **complete_copy_trace_verified=false** and **zero_copy_verified=false**.
Input CPU fallback and validation copies are observed, not inferred zero-copy.
The [Nsight report](cuda-copy/resize-full-copy-004.nsys-rep.gz) and
[exported SQLite](cuda-copy/resize-full-copy-004.sqlite.gz) preserve inspectable
explicit activity. Profiling does not establish Isaac Sim cleaning.

Earlier profiler attempts are retained under [failures](cuda-copy/failures).
They failed component loading because the CUDA test-component overlay was not
sourced. The final run sources both `/opt/ros/lyrical/setup.bash` and
`/buffer-source/install/setup.bash`. The acceptance runner now reports the actual
component error instead of a generic failure. Nsight was copied into the owned
container; host drivers, CUDA and ROS installations were not changed.

The Burger description was installed only in the owned Jazzy fixture from package
`ros-jazzy-turtlebot3-description` version `2.3.6-1noble.20260901.054019` (package
SHA-256 `9875d1a37b54d64605d3444d4cd9e6069845206a6fe21cf0b19918517d674893`).
The fixture Body author now preserves the actual model identity instead of
labelling every supplied URDF as Waffle. This is not a physical cross-Body mission
or automatic unknown-robot integration.

Same-model Codex/Native A/B, causal Memory/KNOW benefit, second-Body physical
cleaning, unknown-Body automatic provisioning, Isaac Sim cleaning, ROS1 physical
navigation and REAL hardware remain unaccepted. No count from replay or unit
tests is substituted for those gates.

## Validation and reproduction

Local ROS/daemon/knowledge-feedback/Practice regression: **474 passed, 9 skipped,
10 deselected**, 38.01 seconds. Memory regression: **192 passed**, 24.34 seconds.
Counts are per run and overlap. Changed Practice adapter type checking passed;
lint passed for `src`, `tests` and the acceptance scripts. Required PR checks must
pass before merging the follow-up. [Regression logs](regression) retain results.

Entry points under `integrations/ros_probe/acceptance`:

- `safety.py daemon_restart`: fresh owned golden world; MCP request_action to a
  separate daemon; `--persistent-ledger` is confined to the daemon fixture.
- `memory_reuse.py`: `--source-memory`, `--verification`, `--output`; actual accepted
  verifier required, source read-only, output directory must be new.
- `knowledge_feedback.py write` then `read` in separate processes: an owned SeekDB
  copy plus actual advice, verification and Practice files; exact immutable
  feedback JSON reused for the second process.
- `second_body_contract.py`: actual `--waffle` and `--burger` vendor URDFs; `--output`.
- `isaac_resize.py`: inside the isolated ROS_DOMAIN_ID=201 GPU fixture. Nsight 2025.3.2
  uses `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
  --trace-fork-before-exec=true --cuda-trace-all-apis=true`. Import with matching
  QdstrmImporter, export SQLite, then run `cuda_copy_trace.py` with `--database`,
  `--validation`, `--output`. Explicit CUDA activity and whole-copy completeness
  are separate results.
