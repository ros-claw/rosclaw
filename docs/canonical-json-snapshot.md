# Canonical JSON snapshots

`growth.canonical_json_snapshot.CanonicalJSONSnapshot` serializes a complete
ordinary JSON dictionary once into owned immutable bytes. `verify()` recomputes
the SHA-256 over those bytes; `restore()` verifies then decodes a freshly owned
dictionary. Nested caller mutation cannot change the snapshot. Non-finite values
and lossy JSON coercions are rejected. Private byte corruption fails closed.

This avoids repeatedly encoding large, already validated numerical documents.
The content hash has the same definition as Growth's canonical JSON hash.
It is identity only, not provenance, a signature, policy validation, physical
evidence, promotion, or authorization. Consumers must still validate their whole
model at allocation, check source pins, and verify their own execution contracts.
Do not replace validation of new caller-supplied models with a cached hash.

No Torch dependency, robot dimensions, action selection, simulator commands, or
hardware access are introduced. A downstream execution optimization additionally
requires complete numerical and native-physics parity before adoption.
