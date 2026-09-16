# SeekDB 1.4.0 Qualification — 09: Release Hardening & Production Acceptance

**Date:** 2026-09-16 · **PR:** PR-SDB-140-5 · **Scope:** the FINAL closing PR —
no new features, only release blockers + production acceptance (per the
2026-09-16 review directive `seekdb升级优化实施0916.md`).

## Support levels (official after this PR)

| Mode | Status |
|---|---|
| `server` + Engine 1.4.0 + pyseekdb 1.4.0.post1 | **Production / Validated** |
| `legacy_embedded` + pyseekdb 1.4.0.post1 | **Compatibility** |
| `local_runtime` + seekdb bindings 1.4.0.dev2 | **Preview / x86_64 only** |

## P0 blockers fixed

### P0-1..P0-4 — instance identity across the whole lifecycle

Neither "port answers" nor "pid alive" is identity (the launcher re-execs;
seekdb binds SO_REUSEPORT so two engines can share one port).  Now:

- `$SEEKDB_HOME/runtime.json` records the verified identity:
  `schema_version / engine_version / pid / process_start_time / binary /
  base_dir / data_dir / port / started_at` (written only AFTER verification).
- `start_1_4.sh`: port-occupancy gate (identity-verified same instance →
  already-running; unknown seekdb or foreign listener or collision → HARD
  FAIL, exit 6); post-launch the REAL engine pid is found by /proc cmdline
  match AND ppid==1 (daemonized) — the launcher shares the cmdline but is
  not daemonized; SQL proof requires `SELECT VERSION()` to carry
  `seekdb-v1.4.0` plus a real round-trip; runtime.json written, and the
  `.rosclaw_seekdb_1_4` stamp is touched LAST — a failed start leaves no
  stamp (L7).
- `stop.sh`: identity from runtime.json, /proc start-time check defeats PID
  reuse, SIGTERM → wait → SIGKILL only after re-verifying identity;
  multiple candidates → `AMBIGUOUS_INSTANCE` + `REFUSE_TO_STOP` (exit 7),
  identity mismatch → refuse (exit 8).  Never guesses.
- `status.sh` / `doctor.sh`: identity-separated verdicts — PROCESS_OK /
  IDENTITY_OK / SQL_OK / VERSION_OK / DATA_DIR_OK / BIND_OK; READY only when
  all hold; collisions are reported (`AMBIGUOUS_INSTANCE`), never hidden.

### P0-5 — lifecycle adversarial tests L1–L10 (real engines, no mocks)

`tests/integration/seekdb/test_lifecycle_identity.py`, all PASS on the
Jetson rig (9 tests covering L1–L10): normal start identity artifacts;
pid file = real engine post re-exec; stale pid file + live engine → stop
adopts verified instance; foreign (non-seekdb) listener → HARD FAIL;
second seekdb same port (SO_REUSEPORT) → start HARD FAIL + doctor
AMBIGUOUS/NOT READY; failed start → no stamp; PID reuse → refuse, victim
process untouched; SIGTERM graceful; SIGKILL + stale runtime state →
recover, data intact.

### P0-6 — installer support matrix is now truthful

Supported = Ubuntu 24.04 amd64+arm64 only (checksums pinned, install +
doctor proven).  ubuntu22.04 / debian12 / debian13 → `experimental`
(require `ROSCLAW_SEEKDB_ALLOW_EXPERIMENTAL=1` AND a recorded checksum).
No `__FILL_ME__` remains.  `tests/storage/test_seekdb_install_matrix.py`
enforces: supported assets must carry real 64-hex SHA256, no placeholders,
asset names must exist in the upstream v1.4.0 release (integration check),
supported/experimental disjoint.

### P0-7/8/9 — local_runtime honesty

- **Unix socket (P0-7):** `connection_options()` returning `unix_socket`
  is now carried end to end — `SeekDBSQLStore(unix_socket=...)` (pymysql
  native) and `SeekDBServerRetrievalStore(unix_socket=...)` (pyseekdb
  **kwargs → pymysql).  The socket IS the instance identity; nothing falls
  through to `127.0.0.1:2881` behind its back.
- **Ownership (P0-8):** `LocalRuntimeStructuredStore` composition owns
  BOTH the runtime and the inner store: connect = runtime start-or-attach
  → inner store built from the REAL connection options → inner.connect;
  disconnect = inner.disconnect → owner closes the engine, attacher
  releases only.  The factory no longer orphans the runtime.
- **Shared instance (P0-9):** `SeekDBLocalRuntime.attach()` /
  `start_or_attach()` / `release()` — a second process attaches to a live
  owner (no second engine, no closing the owner's engine) instead of
  `raise already owned`.

Unit tests: 13 in test_seekdb_runtime.py (incl. attacher-never-closes,
owner-disconnect-closes, socket-options build the socket-bound store,
pymysql kwargs proof) + mode/installer suites — all PASS.

### P0-10/P0-11 — Gate H (real x86 engine CI)

`.github/workflows/seekdb-engine-gate.yml` (ubuntu-24.04 amd64):
- **H1 Server:** pinned deb install (SHA256) → identity start → wire
  VERSION proof → doctor → L1–L10 lifecycle → W2R smoke benchmark →
  migration fixture (sqlite→server→parity) → graceful stop.
- **H2 Local Runtime:** pinned `seekdb==1.4.0.dev2`, real engine process,
  real socket, four query legs, two-process attach, close, reopen,
  persistence — no mocks
  (`tests/integration/seekdb/test_local_runtime_e2e.py`; skips on
  aarch64/without bindings).

## P1 acceptance evidence (live, this Jetson, engine 1.4.0 + post1)

### P1-1 clean wheel install — PASS

`uv build` → fresh venv → `pip install rosclaw-1.2.0[seekdb]` →
pyseekdb resolved to exactly 1.4.0.post1 → real engine connect + round-trip
+ capabilities.deployment == server.

### P1-5 Qwen multilingual production path — PASS

64 curated docs (8 scenarios × zh/en × 4 phrasings) × 24 queries
(zh→zh / zh→en / en→zh) through the REAL production path
(VersionedCollectionManager + Qwen3-Embedding-0.6B @pinned revision,
query-side instruction, ngram analyzer):

| leg | Recall@1 | Recall@5 | MRR | p95 |
|---|---:|---:|---:|---:|
| vector (Qwen) | 1.0 | 1.0 | 1.0 | 28.7 ms |
| BM25 | 0.9583 | 1.0 | 0.9792 | 26.2 ms |
| hybrid RRF | 1.0 | 1.0 | 1.0 | 27.5 ms |

Baseline in report 04 (engine built-in MiniLM): vector Recall@1 0.833.
The production Qwen path clears it on every zh-heavy query.

### P1-6 rollback point — PASS

1.3 embedded source (416 rows) → dump → restore to 1.4 server (parity ok)
→ new writes land on 1.4 → re-open the UNTOUCHED 1.3 source: 416/416 rows
readable.  Upgrade does not damage the rollback point (no physical
downgrade attempted, by design).

### P1-7 network exposure — implemented

doctor's BIND_OK: wide bind (0.0.0.0) → NOT READY unless
`ROSCLAW_SEEKDB_ALLOW_WIDE_BIND=1`.  Verified both ways on the rig.
Finding: the 1.4.0 binary has NO bind-address option (`--parameter
devname=lo` tested — the SQL port still binds 0.0.0.0), so single-node
deploys should firewall port 2881; the check makes the exposure loud
instead of silent.

### P1-2/P1-3 60-min soak + memory reclaim — PASS

Fully scripted campaign (`validation/seekdb/scripts/soak_campaign.sh`,
artifacts in /tmp/soak140_campaign on the rig): identity-verified fresh
restart (dogfooding the new scripts on the data-bearing instance) →
smaps_rollup sampler (15s) → idle 5min → 35k load → 1w4r 60min → 2w8r
60min → idle 10min → idle 30min.

Soak gates: **lost=0, duplicates=0, crashes=0, uncaught errors=0** —
1w4r: 63,361/63,361 writes, 260,747 reads; 2w8r: 127,578/127,578 writes,
465,387 reads.

Resource curve (RSS MiB from /proc smaps_rollup):

| phase | min | max | last | threads last | fds last |
|---|---:|---:|---:|---:|---:|
| idle 5min (fresh) | 600 | 640 | 616 | 85 | 38 |
| load 35k | 636 | 942 | 833 | 100 | 39 |
| soak 1w4r 60min | 760 | 1000 | 800 | 87 | 43 |
| soak 2w8r 60min | 786 | 1314 | 903 | 104 | 48 |
| idle 10min | 756 | 901 | 779 | 82 | 38 |
| idle 30min | 750 | 824 | 754 | 83 | 38 |
| end | 753 | 802 | 768 | 83 | 38 |

Verdict: **no leak** — 2w8r's second-half mean (872) is BELOW its
first-half mean (917); the idle tail reclaims to ~750-770 MiB and stays
flat; threads (78-105 band, ends at 83 vs 85 at start) and fds (38→48
under load → 38) show no growth.  The ~1 GiB figure seen in earlier
sessions is load high-water + cache, reclaimed after load stops.  The
upstream 166 MiB figure is a cold/tuned idle number, not our gate; our
gate (no monotonic growth + reclaim) passes.

### P1-4 35k medium workload — PASS

35,000 records (10k episodes / 5k failures / 20k memory_nodes, ROSClaw
semantics zh/en across UR5e/RH56/LIMO/Nova Carter) into the live server:
load 2441.5s (14.3 rows/s — server-side MiniLM embedding dominates),
query p95 at 35k scale: metadata 2.1ms / BM25 8.6ms / vector 56.3ms /
hybrid 69.2ms; W2R p95 94.8ms visible_fraction 1.0.  Restart persistence:
35,021 rows + marker intact after stop/start (`medium_workload.py`,
release-only; the 400-row benchmark stays the CI smoke).

## Release acceptance

Machine-readable: `release_acceptance.json` (this dir).
