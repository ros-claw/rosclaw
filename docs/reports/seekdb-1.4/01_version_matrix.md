# SeekDB 1.4.0 Qualification — 01: Version Matrix

**Date:** 2026-09-15 · **Platform:** Jetson-class aarch64, Ubuntu 24.04,
glibc 2.39, Python 3.13 · **Branch:** pr-sdb-140-2

## Matrix status after PR-1 + PR-2

| Lane | Engine | SDK | Binding | Result | Evidence |
|---|---|---|---|---|---|
| A baseline | 1.3.x (container) | pyseekdb 1.3.0 | pylibseekdb 1.3.0 | ✅ PASS | baseline_1_3_server.json / baseline_1_3_embedded.json |
| B primary | **1.4.0** (deb, server) | pyseekdb 1.3.0 | — | ✅ PASS | engine_1_4_sdk_1_3_server.json; doctor: `5.7.25-OceanBase seekdb-v1.4.0.0` |
| C extended | 1.4.0 | pyseekdb 1.4.0.post1 | — | ⏳ PR-SDB-140-3 | |
| D negative | any | pyseekdb **1.4.0** | any | 🚫 fail-fast at connect | tests/storage/test_pyseekdb_compat.py |
| local-runtime | 1.4.0 | seekdb 1.4.0.dev2 (experimental) | seekdb-bindings | ⏳ PR-SDB-140-4 | PyPI 只有 dev2，明确按实验路径处理 |

## Installation record (pinned, no `latest`)

| Asset | SHA256 |
|---|---|
| seekdb_1.4.0-100000212026082616ubuntu24.04_arm64.deb (this Jetson) | `1239102f381b0f93b4d1c72bab5101e42b7ab805f12307d4b5a26a5a37af846f` |
| seekdb_1.4.0-100000212026082616ubuntu24.04_amd64.deb (CI runners) | `18f6b329d462a8fd841d43a12574cacce75f12b06156bdb0c7700d782478d5b1` |
| seekdb_1.4.0-100000212026082616ubuntu22.04_arm64.deb | `d20bb4910e7eff5be9e147057e51c6ef5c6cafa9dce697602d03586d7c68bf78` |

Checksums recorded from first-hand verified downloads; `install_1_4.sh`
refuses to install any asset whose checksum is not recorded.

## Engine deployment facts verified first-hand

- The 1.4.0 deb installs `/usr/bin/seekdb` + obshell; `systemctl` unit exists
  but the qualification tooling runs the binary directly under
  `$SEEKDB_HOME` (no system service mutation).
- The launcher re-execs: the pid captured at spawn is the dead wrapper;
  `status.sh`/`doctor.sh` resolve the live pid by port.
- Engine self-reports `5.7.25-OceanBase seekdb-v1.4.0.0` over the MySQL
  protocol.
- **pylibseekdb's C++ atexit handler overrides the process exit code to 0**
  (verified: `sys.exit(1)` after an open/closed embedded store exits 0).
  Any CLI that opens an embedded store must use `os._exit()` for its
  verdict to survive.  This is now documented in migrate_1_3_to_1_4.sh.
