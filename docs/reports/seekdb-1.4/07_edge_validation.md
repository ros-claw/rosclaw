# SeekDB 1.4.0 Qualification — 07: Edge Validation (aarch64)

**Date:** 2026-09-15 · **This machine IS the edge target** (outline §十八/十九
calls for Jetson Orin / DGX Spark; this is a Jetson-class board):

| field | value |
|---|---|
| OS | Ubuntu 24.04 (nvidia) |
| kernel | 6.17.0-1021-nvidia |
| arch | aarch64 |
| glibc | 2.39 |
| Python | 3.13 (project .venv) / 3.12 (qualification scratch venvs) |
| seekdb Engine | 1.4.0 (`seekdb_1.4.0-100000212026082616ubuntu24.04_arm64.deb`) |
| SDK | pyseekdb 1.3.0 (project pin) / 1.4.0.post1 (qualified, PR-3) |
| binding | pylibseekdb 1.3.0 (1.3.0.post3 on the ty1200 lock) / 1.4.0.post1 |
| ROS2 coexistence | N/A on this unit's test plan (no ROS graph running during the matrix) |

## Edge results

- Install: pinned deb, SHA256-verified, `dpkg -i` clean.
- Engine start: READY on :2881 in ~2s wall.
- Regression matrix T0–T15: 16/16 ×3 on the bundled 1.4 embedded engine
  (post1) and on the 1.3 embedded combo.
- Live benchmark + migration + concurrency + kill-9 fault injection: all
  PASS (reports 02/05/06).
- Warm-after-benchmark server footprint: RSS 705 MiB / 86 threads / 38 fds
  (default memory_limit=2G; the official 166 MiB idle figure is cold+tuned —
  see 05's caveat).
- **Boundary found:** `seekdb` bindings wheel is x86_64-only →
  local_runtime is x86_64-only today; aarch64 uses `server` or
  `legacy_embedded`.  This is exactly the kind of edge fact the outline's
  §十八 exists to record.
