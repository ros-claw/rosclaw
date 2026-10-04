# Exact-coordinate index for frozen output memory

`IndexedAnchorOutputMemory` is an opt-in compiled query representation of
`AnchorOutputMemory`. It retains every logical observation/prediction row,
original order, provenance, schema, authority flags and memory hash.

The original accelerated nearest-reference query asks for two neighbors. When
many logical rows contain identical coordinates, this often triggers its
deterministic full-bank NumPy tie path. The new private index stores one copy of
each exact coordinate and maps it to its first original logical index. It does
not subsample, round, quantize, merge distinct states, or discard evidence.
Conflicting predictions at identical states are still rejected by the original
constructor. Signed zero is normalized by that same constructor.

For ties or numerically ambiguous distances between *distinct* coordinates,
the original complete-bank NumPy arithmetic and lowest-index rule remain in
use. SciPy is optional; without it, all queries use the reference path.

```python
from rosclaw.growth.indexed_anchor_output_memory import IndexedAnchorOutputMemory

compiled = IndexedAnchorOutputMemory.from_dict(sealed_logical_memory)
assert compiled.to_dict() == sealed_logical_memory
prediction = compiled.blend(observation, proposal, encoder_hash=frozen_encoder_hash)
```

Compilation is explicitly selected. The base class, existing models and
validators are unchanged. `extend` deliberately returns the original logical
implementation; a caller must explicitly compile the resulting bank again.

Tests compare exact queries and blended outputs with the reference in 1, 12,
135 and 512 dimensions, including duplicates, signed zero, distinct ties,
near-ties, permutations, missing SciPy, schema tampering and authority drift.
Unit-test equivalence is not proof of accelerated end-to-end training or
better physical skills. Simulator consumers need full stored-trace equivalence
and independent dynamics audits before adopting a new compiled execution path.
No runtime, hardware, promotion or distributional-retention authority is added.
