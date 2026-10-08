# Explicit saturated prediction-memory query

`BoundedBlendOutputMemory` is an opt-in numeric view of the complete existing
`AnchorOutputMemory` artifact. Existing defaults and artifact schemas are not
changed. Construction, encoder identity, observation/proposal validation,
duplicate handling, logical serialization and immutable arrays remain checked.

The original blend is `g * proposal + (1 - g) * nearest_prediction`. When the
original float64 gate is exactly one, the prediction has zero weight. For an
all-nonzero finite proposal, the new view can return an owned copy without
searching for that prediction. For a proposal containing **any zero**, it still
executes the original arithmetic: skipping it could change signed-zero bits.
Near anchors and non-saturated gates use the original indexed nearest search
and blend. Gate evaluation uses the separately validated bounded-query view;
without optional SciPy, the full NumPy reference arithmetic remains available.

The entire logical prediction bank, including ordered duplicates, remains in
the serialized artifact. Extension returns the ordinary memory and requires
explicit recompilation. This is not consolidation, forgetting, a new policy,
training, a runtime interface or authorization. A numeric unit-test pass is
not physical replay parity; simulation adoption needs separately source-bound
complete trajectory comparison against the original implementation.
