# ROSClaw Know / How final acceptance

Overall status: PASS for software engineering and real-world knowledge-system
acceptance, with hardware and the unavailable final vLLM paired run explicitly
deferred.

## Acceptance scope

In scope:

- software engineering and package integration;
- real-world project, paper, and official-document knowledge;
- Project Wiki fidelity and evidence closure;
- SeekDB server, migration, full-text, vector, and hybrid retrieval;
- version, provenance, temporal validity, and disagreement handling;
- How advice, compatibility rejection, and structured explanation;
- Native Agent routing and bounded reference workspace;
- security, failure degradation, and packaging.

Out of scope:

- physical-task improvement on Unitree G1, LIMO, RealSense, arms, or hands;
- online autonomous knowledge mutation;
- large-scale multi-node production capacity and availability.

| Gate | Result | Evidence summary |
|---|---|---|
| A — Architecture | PASS | Boundaries preserved; How has no Action authority |
| B — Real source | PASS | 7 GitHub projects, RoboNaldo paper, official docs, fixed snapshots |
| C — Wiki fidelity | PASS | 70/70 critical claims; all severe-error counts zero |
| D — SeekDB | PASS | Server, migrations, BENG/IK/NGRAM, vector, hybrid/RRF, rollback, restart, backup/restore |
| E — Temporal | PASS | A→B, selective rebuild, supersession, old evidence, stale packs, release X→Y |
| F — Native Agent | PASS | 30 boundary cases; 0 Know/Memory and 0 How/Action crossings |
| G — Self-bootstrap | PASS | Two real official-doc packs/advice records produced code and live tests |
| H — Security | PASS | Injection inert; no code, secret, symlink escape, Action, or Memory crossing |
| I — Packaging | PASS | clean build/check/install/reinstall; migrations present; only Know/How published |

Test summary:

- rosclaw-know: 702 passed, 14 opt-in/default skips;
- live knowledge: 7 passed in 106.52 s;
- rosclaw-how: 363 passed;
- Core knowledge: 64 passed;
- rosclaw-agent: 32 passed;
- Know GitHub Actions: lint + Python 3.11 + Python 3.12 passed;
- Core PR #255: every required check passed, including Python 3.11/3.12/3.13,
  Full Regression, Integration Test, Build Package, ROS Docker, Node, lint,
  type check, product acceptance, and boundary checks. Core PyPI release was
  skipped as required.

Performance samples were comfortably inside product SLOs. The in-memory
200-sample p95 values were 0.234 ms for pack build, 0.225 ms for ordinary
consult, 0.226 ms for exact-error diagnose, and 0.004 ms for evidence open.
The real SeekDB native hybrid p95 was 9.35 ms across 20 queries. The fixture
numbers demonstrate regression headroom, not a distributed production load
claim.

ROSClaw now has a versioned world-knowledge layer whose accepted conclusions
are snapshot-bound and evidence-closed. Know can ingest real projects, papers,
and official documentation, distinguish source authority, model temporal
replacement and disagreement, and explain retrieval. How can select or reject
that knowledge against current context and explain its decision without
crossing into execution authority. Native Agent routing and workspace state
make those capabilities available without mixing Body, Memory, Skill, or
Action responsibilities.

No claim is made that this release improves Unitree G1, LIMO, or RealSense
physical task success. Hardware outcomes and online autonomous refresh remain
separate future acceptance stages.

## Exact release manifest

| Component | Accepted version / commit | Immutable artifact |
|---|---|---|
| rosclaw-know | `1.3.0` / `3e15dfb2d07e3ff80d870224dc81f3e4f555fc56` | wheel SHA-256 `ae0a418a1f574449a556b0b17fe9064eb0ba0cff270844b5e514c5f59513ea4d` |
| rosclaw-how | `1.3.0` / `9b866d4c01c5cdfd63cd997f1fee5e61f1817275` | wheel SHA-256 `bc4ab9ee22b8050ab27f9a8a25d4c78ae1e0966a5df78cee34cd1957ceb2d34a` |
| rosclaw Core | `aa124592a1ab038e59a4028483673b47495195ef` | Git tree `75496cc9f2edae4d7d5b6e2a0e4cb4936a691aa0` |

| Contract / runtime | Accepted value |
|---|---|
| Know public-contract schema SHA-256 | `f5372507a9c5dfc9257425a9cb5d932a2c904a5b99a0cf57da96a1ef96a51867` |
| SeekDB migration count / SHA-256 | `7` / `a4d3557150072b5ed2ab5ba8dd7844dd04f6f14050660063f0e358265808585b` |
| SeekDB server | `seekdb-v1.3.0.0` |
| SeekDB image digest | `sha256:e3a46b6520fa6b6fb7949d03b8c6f22cef180e6c84953b839ad56a358d34932d` |
| pyseekdb used by live Know acceptance | `1.4.0.post1` |
| pylibseekdb used by live Know acceptance | `1.3.0.post4` |

## Frozen acceptance bundle

The immutable closure label is `final-acceptance-20260807`. Its machine-readable
manifest binds the accepted commits, schemas, migrations, real-source snapshot
manifest, 70-claim audit, SeekDB capability evidence, live result, self-bootstrap
evidence, package hashes, and report digests. See
[`frozen-acceptance-bundle/README.md`](frozen-acceptance-bundle/README.md).
