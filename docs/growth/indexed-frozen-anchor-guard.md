# Optional coordinate index for frozen-anchor lookup

`IndexedAnchorKernelGuard` is an explicitly selected compiled view of an
`AnchorKernelGuard`. It is task-neutral and exposes no rollout, hardware,
activation, or approval interface. Existing guards do not select it automatically.

The complete logical bank is retained, including duplicate rows, original row
order, signed-zero serialization, provenance in the caller's bank, and the
original guard hash. Only a private search tree stores each distinct coordinate
once. Coordinates are not rounded or clustered. Signed zero is normalized only
in private index keys.

Queries recompute squared distance from coordinate differences with the same
arithmetic as the reference implementation. Distinct-coordinate ties and
numerically ambiguous near-ties use the **original** tree and gate implementation;
the new index never introduces a new tie-breaking policy. Without optional SciPy,
or with `accelerated=False`, the original NumPy path is used. Serialized guard
validation and finite/bounded input validation remain required.

Synthetic tests cover exact output equality, full logical serialization equality,
independent storage, dimensions 1/12/135/512, duplicate and signed-zero coordinates,
ties, near-ties, missing SciPy, and invalid queries. These tests are not evidence
of robot performance. A real-bank benchmark must separately report complete-query
coverage, observed equality, and timings before a caller adopts this optimization.
Neither lookup equality nor local zero gates prove distributional retention,
physical safety, or successful learning. No running experiment is hot-upgraded.
