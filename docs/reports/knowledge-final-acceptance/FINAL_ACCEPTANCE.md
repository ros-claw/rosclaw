# ROSClaw Know / How final acceptance

Overall status: PASS for software engineering and real-world knowledge-system
acceptance, with hardware and the unavailable final vLLM paired run explicitly
deferred.

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
- Know GitHub Actions: lint + Python 3.11 + Python 3.12 passed.

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
