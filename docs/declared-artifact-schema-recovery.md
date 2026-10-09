# Explicit declared-schema recovery

## Snapshot-bound registration

Explicit validation now returns the exact bounded immutable artifact bytes to
registration. The kernel persists those bytes in an exclusive private directory
as a read-only regular file and registers that stored object, not the mutable
input pathname. Schema and artifact writers after validation cannot substitute
unvalidated content. The ArtifactRef path therefore names a snapshot rather than
the source path. Legacy calls omit the optional snapshot argument entirely.
Four additional tests exercise artifact and schema changes, growth and symlink
replacement against actual kernel registration and compare stored bytes and SHA.
This is a source-only software change, not physical or task success.


Only `rosclaw_deliver` with a nonempty explicit `schema_path` opts into content-aware failure suppression. No filename infers a schema. Legacy delivery and other tools retain their existing argument fingerprints.

Idempotency lookup remains first: an old key returns the exact original response even after a repair. Use a fresh key after editing the actual artifact or schema bytes at the same admitted path. A changed pair is fully revalidated; changed-but-invalid content remains a typed rejection with zero task/artifact rows. An unchanged failed pair returns `DOOM_LOOP`. Returning to an earlier failed pair is also blocked.

Content reads happen inside declared validation after the existing session, mission, writer and effect validation chain and workspace path admission, never in the preauthorization fingerprint function. Private SHA-256 digests distinguish the two bounded byte inputs; digests and private property names/values are not included in diagnostics. Validation uses precisely those bytes. Existing vocabulary, finite JSON, depth, node, byte and diagnostic budgets remain unchanged.

Reads walk resolved paths through no-follow descriptors, require regular files, bound each read to cap+1 bytes and reject metadata changes during reading. Permission, replacement-link and I/O failures fail closed with a generic typed rejection. Workspace escape remains rejected. This is software-level validation, not task acceptance or physics evidence; concurrent changes after validation are not evidence of successful delivery.

Four appended native tests cover artifact-only repair, schema-only repair, changed-invalid revalidation and unchanged-byte suppression, including immutable old-key replay and zero-row negative assertions. Original tests are preserved.
