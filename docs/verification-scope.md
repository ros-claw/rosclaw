# Verification scope

New completion receipts retain `verification: PASS` for actual successful checks,
while `verification_scope` states what those checks establish. This field is
persisted with verification checks, events, direct finish responses and task
outcomes before completion is committed.

| Scope | Evidence |
| --- | --- |
| `artifact_integrity_only` | Registered files exist, are nonempty and match their recorded hashes. |
| `summary_nonempty_only` | A nonempty response was recorded. |
| `configured_acceptance` | Frozen required-file, test-command or source-packet checks were executed. |
| `declared_deliverables` | Frozen required deliverable kinds were checked. |
| `configured_acceptance_and_deliverables` | Both explicit check sets were applied. |

Integrity and summary checks carry `task_semantic_verification: UNVERIFIED`.
Configured checks carry `CONFIGURED_CHECKS_ONLY`: passing those checks is limited
to their declared criteria, and does not establish full goal satisfaction or
physical behavior. Unknown acceptance keys and optional deliverables do not
create configured acceptance. Artifacts remain opaque.

Successful final delivery can still complete normally. Progress and diagnostic
artifacts retain their existing non-final behavior. Existing provenance, media,
embodiment and failure checks remain in effect. Terminal presentation distinguishes
delivery completion from semantic acceptance without making normal file delivery
fail. Historical terminal outcomes and verification records are not migrated,
recomputed or backfilled; cached outcomes retain their original bytes.
