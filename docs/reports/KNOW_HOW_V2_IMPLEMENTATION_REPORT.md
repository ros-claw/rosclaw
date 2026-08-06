# Know / How v2 implementation report

Date: 2026-08-06
Target release: 1.2.0
Source plan: `know_how_优化v1.md`

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

The implementation is present as uncommitted working-tree changes based on:

| Repository | Base commit | Package version |
| --- | --- | --- |
| `rosclaw-know` | `55ce4981f46ca469ff1cc6ca9582f82d6e2d8c92` | 1.2.0 |
| `rosclaw-how` | `4639cd68bbe294ce61eb01d63c9a96999e3f969f` | 1.2.0 |
| `rosclaw` | `fbf9a692bbf759fdc8f8e304d205c3d592ca4dc9` | 1.2.0 |

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
bounded process-local advice lookup, feedback, health and capabilities.
Capabilities explicitly state that How owns neither a retrieval index nor
action authority. Advice and feedback endpoints enforce the existing optional
How API-key allowlist. The legacy `/wiki/v1` path can be rolled through the v2
engine behind a feature flag.

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

## Verification performed

All tests used fixtures, mocks, temporary directories or embedded stores. No
real ROS graph, robot, actuator or hardware service was contacted.

| Check | Result |
| --- | --- |
| `rosclaw-know` full pytest | 682 passed, 7 skipped |
| `rosclaw-how` full pytest | 357 passed |
| Core knowledge + E2E + MCP + CLI focused regression | 100 passed |
| Real installed-package in-process Know→Pack→How→feedback loop | passed |
| v2 Ruff scopes and all three `git diff --check` checks | passed |
| Three 1.2.0 wheel builds | passed |
| Wheel schema/migration/module inspection | passed |
| Python 3.13 clean full-dependency wheel install | passed |
| Clean uninstall and `--no-index --no-deps` local-wheel reinstall | passed |
| `rosclaw[knowledge]==1.2.0` dependency resolution | passed |
| `rosclaw[all]==1.2.0` dependency resolution | passed |
| Installed entrypoint disabled/degradation check | passed |
| Core full pytest | 6375 passed, 74 skipped, 39 deselected, 6 environment failures |

The clean wheel environment reported all three installed versions as 1.2.0,
found the packaged JSON Schema, found all 12 migration files and opened an
isolated SeekDB embedded database.

The six Core full-suite failures are outside the changed Know/How areas. Two
external-worker tests assume the `codex` binary is absent, while this host has
an incompatible/untrusted-directory Codex CLI. Four LeRobot integration tests
are selected by their availability probe but the configured worker interpreter
cannot import LeRobot. Focused reruns of all changed Core surfaces pass.

## Known limits and release follow-ups

- SeekDB server-mode construction, SQL migrations and feature negotiation are
  implemented, and Core server-parameter wiring is unit tested, but no live
  server endpoint was supplied; server integration, transaction rollback and
  `AI_RERANK` remain unverified on this host.
- Live GitHub, arXiv, official documentation and external MCP service tests
  were intentionally not run. Default CI remains deterministic and offline.
- Native multi-vector is not available in the probed collection API; the
  implementation uses coordinated vector collections/fields and RRF.
- Advice lookup is a bounded process-local cache, not a durable How database.
  Reference Packs themselves remain durable in Know.
- The HTTP clients do not yet serve a previously cached Reference Pack after
  Know becomes unavailable; callers receive explicit degraded/unavailable
  state instead of a stale pseudo-fresh answer.
- HMAC is the included local bundle signer. A managed deployment should inject
  an asymmetric signer and key-management policy through `BundleSigner`.
- Knowledge feedback is stored and available to refresh/ranking governance,
  but automatic source refresh scheduling is not enabled by feedback alone.
- The Know v2 API implements the operational closed loop, but the plan's
  separate discover/ingest, project-compare and admin endpoints are not all
  split into independent public routes yet.
- No physical execution was attempted. Real-hardware validation must remain a
  separately approved ROSClaw workflow through the normal safety chain.

These limits should prevent declaring every checkbox in the source plan's
Definition of Done complete. The implemented default path and the tested
closed loop are usable; server/live-source and deployment-signing validation
remain release gates for environments that require them.
