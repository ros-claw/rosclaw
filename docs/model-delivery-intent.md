# Explicit model delivery intent

`rosclaw_deliver` requires a role before reading or registering the file or
admitting a task. Missing/blank/non-string roles return `DELIVERY_ROLE_REQUIRED`;
unknown strings return `DELIVERY_ROLE_INVALID`.

For intermediate work, including source preparation, use `progress_report` or
`diagnostic_failed_attempt`. `progress`, `diagnostic`, and their underscore
suffix namespaces register evidence and keep the task open. Existing namespace
matching remains case insensitive and ignores surrounding whitespace.

Final roles are explicitly `report`, `plot`, `image`, `video`, and `data`.
They trigger the existing Task Coordinator acceptance flow. The public PI schema
also limits these values: primitive-to-string coercion must not let `null`, `1`,
or `false` acquire final-delivery semantics. A role describes intent; it does not
create media lineage, demonstrate physical behavior, or replace acceptance gates.

New model calls that omitted a role or used another final label must choose one
of these roles. Legacy `rosclaw_artifact_register`, capability automatic
registration, persisted history, and replay remain unchanged. Same-file hash
idempotence does not acquire a progress-to-final promotion feature. An explicitly
final delivery under an empty acceptance contract still follows existing
Coordinator behavior; source-only prose is not automatically interpreted.
