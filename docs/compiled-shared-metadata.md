# Prediction and evidence preparation: shared metadata

`growth.compiled_shared_metadata.CompiledSharedMetadata` owns canonical bytes
of one ordinary shared JSON payload. It checks the unchanged envelope seal,
payload identity, exact marker path, whole reconstructed logical document hash,
and optionally a root document seal such as `report_hash`.

The streaming reconstruction uses the same JSON canonicalization as
`shared_proof_payload`, including Unicode, floats, sorted keys and separators.
It never substitutes strings by text search. Payload bytes and dependency
sources are checked on every call; input/result mutations do not change the
owned payload or subsequent metadata.

The result is explicitly metadata-only and still contains the payload marker.
It is **not** a complete numerical policy, independent physical verification,
proof of provenance, valid Permit, or execution authority. Consumers requiring
policy reconstruction must keep using the original full-payload reader.
The root seal is checked only when explicitly requested and must be outside
the detached payload branch. No existing reader or safety boundary is changed.

This is intended for repeated authentication of large shared learning/evidence
artifacts before data preparation. Performance improvement requires a measured
comparison on the actual workload; unit tests alone prove no speedup or robot
skill gain. Thirteen new tests pass; the Growth regression suite passes 479
tests. A pytest configuration warning in the isolated Soccer Python environment
does not establish a full repository CI pass.
