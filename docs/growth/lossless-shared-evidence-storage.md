# Lossless shared binary storage

`growth.shared_blob_store.publish_readonly_blob` is a task-neutral POSIX data
primitive. It preserves every byte while allowing several **new** logical
artifact paths to share one read-only inode. It does not load policies, operate
simulators, register executors, establish provenance, or grant promotion.

The caller provides an existing serialized source, an exact SHA-256 digest, a
positive byte bound (at most 16 GiB), an existing local store directory, and a
new same-volume target. Publication owns a private copy; it never changes the
source's permissions or contents. The temporary is flushed, made read-only,
and fsynced before atomic hardlink publication. Existing targets always reject;
existing writable, corrupt, or symlinked blobs reject without repair. No copy
fallback conceals a cross-volume target. Only this call's private temporary is
removed during cleanup.

This is not an archival migration: applications must not convert or delete old
evidence to obtain storage savings. They must retain independent trajectories,
receipts, world definitions, model proofs, and labels for each new execution.
Source identity and physical evidence must still be checked independently.
Read-only permission bits and content hashes are not signatures, operator
approval, or a guarantee against a privileged local attacker.

The Soccer consumer serializes a fresh MuJoCo world into its own temporary and
opts into this primitive only for new SIM evidence. Its separate JSON transport
reuses the existing `shared_proof_payload` codec: all numerical fields and the
original logical report seal are restored before normal evidence validation.
Neither optimization changes physics, rewards, observations, action limits,
learned weights, validation gates, or default native execution.

Validation: 15 focused contracts cover exact bytes, independent source storage,
read-only targets, concurrent deduplication, no overwrite, corrupt/writable
blob refusal, symlink refusal, digest drift and byte budgets. The complete
Growth suite is 266 passed. These tests are data-integrity evidence, not a
learning or robot-performance result.
