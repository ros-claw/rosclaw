# Know / How v2 implementation report

Date: 2026-08-06
Baseline releases: rosclaw-know 1.2.1, rosclaw-how 1.2.0
Published hardening releases: rosclaw-know 1.2.2, rosclaw-how 1.2.1
Source plan: `know_how_优化v2.md`

## Result

The three repositories now implement the v2 ownership split and a tested
external-source-to-advice loop:

```text
pinned external material
  -> SourceRecord / immutable SourceSnapshot / EvidenceRef
  -> Project Wiki / KnowledgeUnit
  -> SeekDB-backed hybrid retrieval
  -> ReferencePackV2
  -> context-aware HowAdviceBundleV2
  -> read-only Native Agent / MCP presentation
  -> KnowledgeUsageFeedbackV1
```

Physical execution was deliberately not added to this path. Advice cannot
create an `ActionEnvelope` or bypass Core policy, safety, daemon or driver
boundaries. Memory remains a separately labelled robot-experience input and
is not used as Know's corpus or store.

The v1 foundation was merged and released before this hardening pass:

| Repository | Merged baseline | Package version / state |
| --- | --- | --- |
| `rosclaw-know` | `3e81b4c` | 1.2.1 published |
| `rosclaw-how` | `d2dea03` | 1.2.0 published |
| `rosclaw` | `44d83d1e` | merged; no Core PyPI release requested |

The usefulness hardening is merged across all three repositories. Packages
were published only for Know and How, as requested:

| Repository | Pull request | Merge commit | Final state |
| --- | --- | --- | --- |
| `rosclaw-know` | [#5](https://github.com/ros-claw/rosclaw-know/pull/5) | `ac4c2dd` | [1.2.2 tagged and published](https://pypi.org/project/rosclaw-know/1.2.2/) |
| `rosclaw-how` | [#3](https://github.com/ros-claw/rosclaw-how/pull/3) | `d52ecfd` | [1.2.1 tagged and published](https://pypi.org/project/rosclaw-how/1.2.1/) |
| `rosclaw` | [#252](https://github.com/ros-claw/rosclaw/pull/252) | `b8cec725` | merged; no Core PyPI release |

Core pull-request CI completed all 18 checks successfully. Its conditional
`Release to PyPI` job was skipped; no Core package was produced or published.

## Architecture and contracts

The authoritative contracts live in `rosclaw-know`; strict wire-compatible
copies in How and Core prevent either consumer from importing Know internals.
Schema-drift tests compare those copies to the authoritative JSON Schema.
Unknown fields are rejected, timestamps must be timezone-aware, enumerations
are bounded and wire versions are exact.

Implemented public contracts:

- `ResearchRequestV2`
- `SourceRecordV2`
- `SourceSnapshotV2`
- `EvidenceRefV2`
- `ProjectCardV2`
- `KnowledgeUnitV2`
- `ReferencePackV2`
- `HowAdviceBundleV2`
- `KnowledgeUsageFeedbackV1`
- explicit Body, Software, Runtime and labelled Memory context contracts

The generated `know_how_v2.json` is shipped inside the Know wheel. The full
Core ownership and runtime-mode description is in
`docs/architecture/KNOW_HOW_V2_INTEGRATION.md`.

## rosclaw-know

### Canonical store

`KnowStore` is a light Protocol with explicit capabilities. The production
implementation uses SeekDB; the in-memory implementation is test-only and
requires explicit opt-in. Store creation has no silent production fallback.

Six repeatable up/down migration groups cover sources, snapshots, documents,
wiki pages, units, relations, packs, feedback, research jobs, index versions
and retrieval indexes. They are packaged in the wheel. Snapshot writes are
immutable and content-derived writes are idempotent. Startup isolation guards
reject equal Know, Memory and Practice database names or embedded paths.

Retrieval implements exact error/symbol match, metadata filtering,
full-text/NGRAM branches, coordinated problem/mechanism/content/code vectors,
RRF fusion, relation expansion, deterministic fallback, compatibility
warnings, score breakdown, token budgets and continuation cursors. Index
version records retain vector dimensions and capability state.

The environment had `pyseekdb 1.4.0.post1` and `pylibseekdb 1.3.0.post4` in
the integration environment. The isolated capability probe confirmed embedded
client creation, collections, namespaces, full-text, sparse vector, hybrid
search and RRF. The wheel-installed embedded store also opened successfully
and returned statistics. Native single-collection multi-vector and server
`AI_RERANK` were not advertised; coordinated collections and deterministic
rerank fallback are used instead.

### Sources, wiki and safety

Read-only, bounded adapters exist for GitHub, an allowlisted official-docs
catalog and arXiv. DeepWiki, GitMCP, Context7 and general web adapters report
explicit unavailable state until their services are configured. Default tests
use fixed transports and do not depend on the public internet.

GitHub ingestion pins a commit, inventories repository metadata and relevant
source/config/docs/test/deployment files, caps counts and sizes, rejects unsafe
paths and binaries, and never executes repository content. Prompt-injection
signals are retained as evidence metadata; source text never becomes a system
instruction or tool call.

The wiki compiler creates inventories, language-aware symbols, dependency
relations, project/component/page records, evidence-linked knowledge units and
content-hash incremental updates. Optional `.rosclaw/know.yaml` steering is
data-only and cannot grant execution authority.

### Service and compatibility

The v2 service exposes health, capabilities, schema, research planning/run,
source/snapshot/project/wiki/evidence reads, retrieval/Reference Pack building
and governance feedback. Content endpoints enforce the optional
`ROSCLAW_KNOW_API_KEYS` allowlist. Existing `/know/v1` behavior remains. JSON
bridge and pattern assets can be imported without invented provenance and
exported deterministically.

Offline cognitive-wiki bundles have deterministic ZIP metadata, a manifest,
per-file SHA-256/size verification, path and expansion limits, index/freshness/
license fields and a signer Protocol. The included HMAC signer is for local
offline integrity; deployment can provide a stronger signer. Offline imports
remain explicitly marked as offline and freshness-limited.

### Feedback governance hardening

Every accepted `KnowledgeUsageFeedbackV1` now creates a durable,
deterministic `FeedbackGovernanceRecordV1`. The mapping is conservative:
useful and irrelevant verdicts record signals only; stale enters source-refresh
review; incompatible enters compatibility review; misleading enters downweight
review; unknown enters manual review. Every record fixes
`automatic_mutation_allowed=false`. The API returns the governance result and
exposes a bounded/filterable queue. Idempotent feedback IDs stay idempotent and
conflicting reuse is rejected.

## rosclaw-how

How accesses Know only through the versioned Reference Pack/feedback client;
it owns no SeekDB index. HTTP, in-process and disabled clients have the same
boundary.

DISCOVER, CONSULT, DIAGNOSE and CATALYZE produce advisory recommendations with
knowledge-unit and evidence citations. A citation guard drops invented or
out-of-pack references. Empty, unavailable or invalid Know results cause an
honest abstention rather than a blank pseudo-answer. Compatibility warnings
and unknown context are surfaced explicitly. Know evidence and Memory evidence
are separately labelled in diagnosis output.

The v2 API now includes the four explicit mode endpoints, generic advice,
durable bounded advice lookup, feedback, health and capabilities.
Capabilities explicitly state that How owns neither a retrieval index nor
action authority. Advice and feedback endpoints enforce the existing optional
How API-key allowlist. The legacy `/wiki/v1` path can be rolled through the v2
engine behind a feature flag.

Advice records live in a bounded SQLite store with creation/expiry and
feedback status, so lookup survives process restart. Service-mode Reference
Packs use a separate bounded SQLite cache. Fresh, cached and stale states are
strict contract fields propagated into How advice. An upstream stale label is
preserved; missing or over-age cache causes abstention.

## Core rosclaw

Core contains contracts, Protocols and adapters only. It does not copy the
source, wiki, retrieval or advice algorithms. `disabled`, `service` and
`inprocess` modes are available; disabled is the rollback-safe default.
Optional packages are imported lazily, service clients do not start processes,
and missing packages/services become truthful degraded health rather than a
Core startup failure.

In-process mode gives the Know package its own explicitly configured store and
passes How only a Reference Pack client. It never passes Memory's store.
Body/Software/Runtime context projection rejects raw sensor streams,
trajectories, video and secrets. The feedback adapter carries only knowledge
governance fields. EventBus output is allowlisted to IDs, versions, counts and
statuses.

Native Agent and minimal MCP expose four S0 tools:

- bounded Know research;
- Reference Pack build;
- Reference Pack open;
- How advice.

The research worker has only the bounded research capability. Runtime and
dashboard status include v2 health. The four-mode CLI is available through the
normal `rosclaw` entrypoint while unknown/older subcommands remain on the
legacy path.

## Reference study

Read-only reference clones were locked before implementation. No reference
code was copied or executed. The adopted ideas were bounded repository
inventory and leaf-first wiki generation (CodeWiki), perspective-led research
and citations (STORM), progressive read-only access (GitMCP/GitHub MCP/
DeepWiki), version-aware documents (Context7), typed relations (GraphRAG),
incremental updates (LightRAG) and the actual pyseekdb client capabilities.

Exact revisions and non-adoptions are recorded in
`rosclaw-know/docs/references/KNOW_HOW_REFERENCE_LOCK.yaml` and
`KNOW_HOW_REFERENCE_STUDY.md`. The supplied RepoMaster repository URL was not
available and no substitute was silently selected.

## Usefulness evidence

The reproducible campaign under
`docs/reports/validation_runs/know-how-usefulness-v2/fixture-20260806/`
uses three controlled fixtures aligned with the requested G1 football,
RealSense ARM and LIMO ROS1 lines. Each corpus contains an exact lexical match
that is incompatible with the runtime and a pinned compatible route.

| Metric (3 tasks) | keyword-only control | Know→Pack→How |
| --- | ---: | ---: |
| wrong initial routes | 3 | 0 |
| compatibility errors | 3 | 0 |
| references read before first compatible route | 6 | 3 |
| pinned evidence presented | 0 | 3 |
| oracle-compatible route passes | 0 | 3 |

A paired vLLM run used `deepseekv4`, temperature 0.0 and the same three
tasks/runtime contexts. The treatment changed only by adding the Reference
Pack and How advice. The control found 0/3 exact fixture files and 0/3 expected
routes; treatment found 3/3 and 3/3. Raw outputs are retained, including the
control arm's guessed paths and unverified command suggestions. The prompt
leakage found during the first audit was removed and that invalid artifact was
discarded before the recorded run.

This proves value level 3: cited/context-filtered knowledge changes the next
engineering action and avoids a known incompatible route. It does not prove
physical value level 4 or cross-task accumulated value level 5.

## Verification performed

All tests used fixtures, mocks, temporary directories, SHADOW paths or
embedded stores. No real ROS graph, robot, actuator or hardware service was
contacted.

| Check | Result |
| --- | --- |
| `rosclaw-know` full pytest | 689 passed, 7 skipped |
| `rosclaw-how` full pytest | 361 passed |
| Core `tests/knowledge` | 24 passed |
| contract/schema/governance focused regression | 47 passed |
| cache expiry, restart and upstream-stale boundary tests | passed |
| deterministic three-task usefulness A/B assertions | passed |
| paired vLLM A/B (six requests) | passed; exact file and route 0/3 → 3/3 |
| 1.2.2 / 1.2.1 wheel builds and Twine metadata checks | passed |
| official PyPI JSON/Simple Index and exact-version install smoke | passed for both packages |
| clean Python 3.13 no-deps wheel install + v2 SQLite smoke | passed |
| changed-scope Ruff and all three `git diff --check` checks | passed |
| Core full pytest | 6371 passed, 74 skipped, 39 deselected, 11 environment failures |

The 11 Core failures are outside the changed knowledge surfaces: one offline
release installer lacks its cached `aiohttp` dependency; two tests assume
the host `codex` binary is absent; four LeRobot tests select an interpreter
that cannot import LeRobot; and four firstboot tests expect console scripts in
PATH. The changed Core knowledge suite passes 24/24.

## Known limits and deployment follow-ups

- SeekDB server-mode construction, SQL migrations and feature negotiation are
  implemented, and Core server-parameter wiring is unit tested, but no live
  server endpoint was supplied; server integration, transaction rollback and
  `AI_RERANK` remain unverified on this host.
- Live GitHub, arXiv, official documentation and external MCP service tests
  were intentionally not run. Default CI remains deterministic and offline.
- Native multi-vector is not available in the probed collection API; the
  implementation uses coordinated vector collections/fields and RRF.
- How persistence is local bounded SQLite. Multi-replica deployments must use
  a shared/replicated store or accept per-replica advice/cache visibility.
- HMAC is the included local bundle signer. A managed deployment should inject
  an asymmetric signer and key-management policy through `BundleSigner`.
- Feedback creates refresh/review candidates but intentionally does not run a
  refresh worker or mutate ranking/knowledge without a separate review action.
- The Know v2 API implements the operational closed loop, but the plan's
  separate discover/ingest, project-compare and admin endpoints are not all
  split into independent public routes yet.
- No physical execution was attempted. Real-hardware validation must remain a
  separately approved ROSClaw workflow through the normal safety chain.

These limits should prevent declaring every checkbox in the source plan's
Definition of Done complete. The implemented default path and the tested
closed loop are usable; server/live-source and deployment-signing validation
remain deployment gates for environments that require them.

## Validation intentionally left blank

| Requested real-world check | Result |
| --- | --- |
| live SeekDB server, SQL rollback, AI_RERANK |  |
| live GitHub/arXiv/official-doc sources |  |
| ten real-project wiki audit |  |
| real G1 football implementation |  |
| real RealSense ARM integration |  |
| real LIMO ROS1 integration |  |
| physical before/after success metrics |  |

These cells are blank rather than converted into fixture claims.

## v2 review answers

1. Key sources in the recorded campaign are pinned fixture snapshots listed in
   `source_manifest.json`; no live source is claimed.
2. The Reference Pack changed all three first routes from incompatible lexical
   matches to context-compatible routes.
3. Compatibility filtering avoided ROS1/ROS2, robot and architecture/version
   mismatch candidates in the controlled corpus.
4. Every treatment opened one exact file backed by an evidence ID, snapshot ID
   and content hash.
5. The code changes are durable AdviceStore, explicit cache degradation and
   non-mutating feedback governance.
6. Package, contract, API, failure-boundary and Core knowledge tests pass.
7. Advice/cache reuse is proven across SQLite reopen; cross-task real-project
   reuse is not claimed.
8. Incremental source refresh was not run against a live repository.
9. Feedback enters auditable signal/refresh/review queues and cannot
   automatically mutate knowledge.
10. In the controlled A/B, first-compatible reads fall 6→3 and wrong first
    routes fall 3→0; no real elapsed-time or physical rework claim is made.
