# Shared proof payload contract

`rosclaw.growth.shared_proof_payload` is a task-neutral, pure JSON helper.
Repeated large model proofs may be persisted once by an application without
dropping any numerical fields from the logical evidence document.

`detach_payload(document, location)` returns an envelope and the complete
dictionary payload. `restore_payload(envelope, payload)` verifies the envelope
seal, whole payload identity, exact marker and complete reconstructed document
hash. Neither function opens a file, commands a robot, registers an executor,
changes a policy, evaluates physics or grants promotion.

Locations contain one to eight explicit dictionary keys. A payload reference
is only a canonical SHA-256 identity, never a file path or URL. The application
must restrict persistence to its own evidence store, publish atomically without
overwrite, reject missing/ambiguous/symlinked objects as appropriate, and verify
the original document's receipts/proofs after reconstruction. References are
not recursively followed. Nonfinite JSON is rejected.

Hash identity is not an authenticated signature or a claim of physical success.
This format neither upgrades unsigned evidence nor lowers any acceptance gate.
Old documents need not be converted, rewritten or deleted. A storage migration
requires an independently checked lossless roundtrip, and physical workflows
also need an actual same-policy transport-equivalence experiment.
