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

## Saturation-aware optional index

`RadiusIndexedAnchorKernelGuard` is a second opt-in implementation. Its exact
tree query is bounded at 16 bandwidths: beyond that radius the original float64
formula `-expm1(-d**2 / (2 * bandwidth**2))` already rounds to exactly one
(exponent at most -128). No nearest anchor identity is needed for this saturated
gate. Within the radius, the same coordinate-difference arithmetic, original
zero tolerance, and ambiguous-tie fallback apply. This is not approximate nearest
neighbour search, rounding, a larger bandwidth, or a new protection threshold.

The motivation was a negative performance finding in a real-bank comparison:
the unrestricted two-neighbour unique index accelerated novel training queries
but slowed exact protected-anchor queries by searching for a distant second
coordinate. Both categories must be benchmarked; faster novel queries alone are
not sufficient for adoption. Synthetic radius tests include smooth gates,
saturation boundaries and adjacent float values, nearby ties, single unique
coordinates, optional dependency fallback, and bandwidths 1e-4 through 1000.
