# Binary Cloud Quality (Additive, Realistic Size)

`rosclaw.perception.binary_cloud_quality` is the additive Stage B companion to
`rosclaw.perception.cloud_quality`. The legacy 16x16 / 17-count
`analyze_cloud` API and algorithm are immutable; this module adds a bounded,
realistic-size binary path as **pure generated software only** — no ROS/DDS,
no neural network, no hardware, no Artifact resolver/catalog integration.

## API

```python
from rosclaw.perception.binary_cloud_quality import (
    analyze_binary_cloud,  # (metadata: dict, data: bytes) -> result dict
    analyze_cloud_file,  # (metadata, data_path, *, allowed_root, expected_sha256)
)
```

CLI example (stdout is exactly one result JSON object):

```bash
python -B examples/perception_binary_cloud_quality.py \
    --metadata ABS_JSON --data ABS_BIN --root ABS_OWNED_ROOT --sha256 64HEX
```

## Metadata contract

Exactly seven keys: `width`, `height`, `point_step`, `row_step`, `fields`,
`is_bigendian`, `is_dense`.

- `width`/`height`: plain ints 1..2048 each, product ≤ 307200 points.
- `point_step`: plain int 1..256.
- `row_step`: positive plain int with `height * row_step ≤ 8 MiB`.
- `fields`: plain list ≤ 32 entries; each a plain dict with exactly
  `name`/`offset`/`datatype`/`count`; `name` plain str ≤ 64 UTF-8 bytes;
  `offset`/`count` plain nonnegative ints; `datatype` 1..8.
- `is_bigendian`/`is_dense`: exact `bool`.

## Bounded admission vs quality verdict

- **Admission** (`TypeError`/`ValueError`, raised before any conversion,
  copy, or decode): non-plain types (subclasses, custom objects with
  `__bytes__`/`__int__`/`__iter__`/`__float__` hooks are never invoked),
  unknown or missing metadata keys, oversize dimensions, oversize field
  lists or names, oversized row envelopes, non-`bytes` or > 8 MiB data.
  Diagnostics are short (≤ 512 bytes) and never echo caller type names,
  keys, or values.
- **Quality verdict** (`QUALITY_INVALID`, counts `total/0/total`, null
  statistics): admitted-but-malformed layouts — duplicate field names,
  invalid xyz field layout (xyz must each be exactly one FLOAT32
  `datatype=7`, `count=1` field fully inside `point_step`), zero field
  count, stride shorter than `width * point_step`, or data length not equal
  to `height * row_step`.

## Result

Exactly seven keys: `status`, `total_count`, `valid_count`,
`invalid_count`, `min_xyz`, `max_xyz`, `centroid_xyz`.

- `status ∈ {VALID_CLOUD, NO_VALID_RETURN, QUALITY_INVALID}`.
- `total_count` is the honest declared `width * height`;
  `valid_count + invalid_count == total_count`.
- A point is valid only if all of x/y/z are finite. No valid points yields
  `NO_VALID_RETURN` with null statistics.
- The centroid is computed with `math.fsum` over the valid coordinates, so
  cancellation-heavy finite float32 values (e.g. `[2**80, 1, -2**80]`) match
  an independent `math.fsum(values)/n` within 1e-10. The accepted numeric
  domain is unchanged: every finite float32 coordinate is admitted.

## Bounds

| Bound | Value |
|-------|-------|
| Max points | 307200 (e.g. 640x480 or 32x32) |
| Max binary bytes | 8 MiB |
| Max fields | 32 |
| Max field-name UTF-8 bytes | 64 |
| Direct diagnostic | ≤ 512 UTF-8 bytes |
| CLI metadata JSON | ≤ 16 KiB |

## File scope

`analyze_cloud_file` admits only an ordinary regular file that resolves
inside the explicitly caller-supplied `allowed_root` (no root escapes, no
symlinks, no special files). Metadata admission completes before any
payload read. After the root-containment check, every **raw** component of
the caller's data path is walked with `lstat` — crucially *before* any
lexical `.`/`..` collapse (`abspath`/`normpath` would erase a
`link/../file` symlink component before it is ever inspected). A symlink in
the leaf **or in any parent directory component** rejects the path before
any payload read, even when the link destination stays inside
`allowed_root` (e.g. `dir_link/../cloud.bin` whose canonical target is
in-root). An unreadable or unresolvable raw component (e.g.
`missing/../cloud.bin`, where the canonical target exists in-root but the
raw `missing` component does not) likewise rejects before any payload
read: the walk never stops early and trusts the normalized destination.
Plain `..` between ordinary directories (`ordinary/../file`)
remains admitted: dotdot alone is never a rejection cause. Ordinary
nested paths inside the root remain admitted. Pre-open envelope guards use
`lstat` to reject symlinks,
non-regular files (FIFO/socket/device — no blocking open ever happens), and
oversize files promptly; size is re-checked via `fstat` before a bounded
read, and `expected_sha256` must match the current contents exactly.
Rejection diagnostics are fixed short messages (≤ 512 UTF-8 bytes) that
never echo the caller's path. No adversarial atomic-snapshot or universal
memory/wall-time guarantee is promised. Never read binary cloud data as
text and never inline a full cloud as a JSON list.
