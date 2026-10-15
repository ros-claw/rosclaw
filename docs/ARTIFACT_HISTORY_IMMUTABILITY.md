# Ordinary artifact history: SOURCE-only change

## Public storage policy

New ordinary TaskKernel registrations read the source once and own a separate
private snapshot (exclusive directory, exclusive file, fsync, mode 0400).
`artifacts.path` and the returned Ref path identify that snapshot, not the
mutable workspace source. Basename, media type and original run-zone attribution
are preserved. Kernel-written `metadata_json.source_path` records original source
provenance; `payload_storage=owned_snapshot_v1` describes the new protocol.
Caller metadata cannot override either field. Source provenance is not a readable
historical payload locator. CLI list exposes `source_path`; open/show/path label
it as provenance only. `artifact path` prints ONLY the absolute immutable
payload path on stdout, with no provenance line.

Ordinary binary/video files do not inherit the declared JSON 256 KiB bound.
Declared-schema validation and its bounded validated-byte snapshot stay separate;
registration does not reread the unvalidated source in place of those bytes.
Scratch rejection, evidence producer restrictions and model EXPERIMENTAL marking
remain. Snapshot ownership is file/inode separation, not a security claim against
a hostile process with the same UID. Read-only modes are not cryptographic seals.

## Replay identity

Idempotent replay requires equal task, revision, SHA, role, producer,
producer-operation, media type and zone. It verifies the previous bytes before
returning the same Ref without changing its metadata. A different role, revision
or producer identity gets a distinct correctly attributed Ref, never inherited
historical delivery/trust context. Metadata cannot elevate producer identity.
Legacy unowned replay fails explicitly rather than presenting mutable storage as
a newly owned object. No schema migration or historical row rewrite occurs.

## Projection and export

Projection validates stored SHA and size, then writes independent copies with
exclusive creation. It never hardlinks authoritative snapshots. Original basename
is preferred; name collisions use the existing artifact-ID prefix convention.
Evidence stays out of outputs; only current-revision rows are considered.
Changed aliases cannot mutate a snapshot. Invalid snapshots degrade projection
without producing their bytes. Registration still succeeds if the view degrades.

CLI export opens every payload component without following symlinks, requires a
regular file, and verifies both size and SHA, including equal-length corruption.
It verifies before touching a destination. Destination directory traversal also
rejects symlinks. A complete independent temporary copy is exclusively published;
existing files and dangling symlinks are not overwritten. Missing/corrupt/unsafe
payloads return nonzero without creating the destination or its parent tree.
An export to an existing directory retains the original filename.

## Legacy and limits

Legacy rows can still export only when their currently stored bytes verify
against the original ledger size and SHA. Already-overwritten or missing legacy
bytes remain unresolved. A SHA is not a backup: there is no automatic recovery,
reconstruction, path rebinding or migration to invented historical bytes.
No cleanup/GC policy is introduced; snapshots persist with their ledger references.
Metadata-only server resolve is unchanged: stronger live resolve/open/path tamper
validation is NOT claimed. Export and projection are the verified byte consumers
in this narrow change; raw same-UID access can still tamper with local files.

## Verification boundary

New Python regressions exercise equal-size UTF-8/binary history under in-place
write, atomic replacement, deletion and samebytes detach; contextual replay and
actual revision creation; projection inode isolation and pre-copy verification;
legacy/current corruption, missing files and source/destination symlinks; binary
payloads above the declared cap and zone guards. New Node integration invokes
actual `buildWorkspacePackTools` write/edit and real Python TaskKernel/SQLite/CLI
subprocesses in owned temporary trees, both with and without external detach.
No replacement writer, kernel or exporter is used. It checks old/new digests,
exports, source deletion, alias tamper and cleanup. Existing schema, delivery,
revision and CLI regressions are retained, with only the scoped list assertion
updated to verify immutable bytes plus original-source provenance and artifact ID.
The approved first-delivery path assertion alone now checks immutable bytes and
source provenance for both relative and absolute cases; workspace/state/task ID
assertions remain unchanged. The capability-render test remains unmodified and
NOT_RUN in the NO_ENGINE selection, not skipped or counted as PASS.

Replay now normalizes absent zone metadata to the empty-string outside-run zone
returned by `zone_of`; otherwise identical ordinary registrations really replay.
The shared payload verifier replaces baseline `DECLARED_SNAPSHOT_REPLAY_*`
errors with `ARTIFACT_PAYLOAD_*` errors for unavailable/unsafe/changed replay
bytes. This is an error-code compatibility change, not a claim of unchanged
error names. Declared validated-byte isolation, bounds and pre-admission schema
rejection remain; no frozen schema test was changed to accept new error names.

Only the bound SOURCE CHECK and RUN receipts determine tested status. This is
not runtime adoption, a provider/engine test, legacy recovery, dynamics simulation
or robot success. Publication and production adoption require separate review.
