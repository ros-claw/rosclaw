# Temporal update report

Status: PASS for the software update path; no claim of autonomous live refresh.

The controlled commit A→B test verified:

- a new immutable SourceSnapshot is created;
- the old snapshot and old evidence remain queryable;
- only page types using the changed component are rebuilt;
- unchanged deployment/overview pages are not rebuilt;
- same-source changed claims receive `status=superseded`, `valid_to`, and
  `superseded_by`;
- Reference Packs using A are marked cached/stale with a reopen warning;
- the refreshed pack uses B.

The release X→Y test compiles `MigrationNote`, `DeprecatedAPI`, and
`CompatibilityConstraint` from release text and traces supersession from the
old release claim to the new release claim.

Cross-source disagreement follows a different path: neither source is silently
superseded. Both claims remain active but unresolved, a `CONTRADICTS` relation
and pending SourceDisagreement record are created, and Reference Pack use is
rejected until explicit review.

`rosclaw-know refresh` is dry-run by default. `--apply` is required for any
mutation. Feedback and disagreement review commands record decisions but do
not silently rewrite source truth.
