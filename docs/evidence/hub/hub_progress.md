# ROSClaw Hub Implementation Progress Report

**Date:** 2026-06-22  
**Scope:** ROSClaw Hub subsystem (`src/rosclaw/hub/`, `tests/hub/`, `docs/hub/`)  
**Status:** Phase 1–6 implemented, documented, passing tests, and merged to `main` via PR #18.

## Summary

The ROSClaw Hub subsystem turns the local runtime into an asset-aware platform
that can discover, validate, install, update, uninstall, and publish physical-AI
assets from a registry. It supports five asset types: `skill`, `provider`,
`hardware_mcp`, `digital_twin`, and `cognitive_wiki`.

## What was delivered

### Phase 1: Schema, refs, fixtures, validate / ref-parse CLI

- `src/rosclaw/hub/__init__.py`
- `src/rosclaw/hub/errors.py` — `HubError`, `HubErrorCode`
- `src/rosclaw/hub/refs.py` — `AssetRef`, `parse_ref()`, `ref_from_dict()`
- `src/rosclaw/hub/schema.py` — Pydantic v2 manifest models + JSON Schema export
- `src/rosclaw/hub/cli.py` — Hub subcommand registration and dispatch
- Tests: `tests/hub/test_schema.py`, `tests/hub/test_refs.py`
- Fixtures for all five asset types under `tests/fixtures/hub_assets/`

### Phase 2: Cache, index, fake registry, sync / search CLI

- `src/rosclaw/hub/cache.py` — `HubCache` with blobs, manifests, installed state, staging
- `src/rosclaw/hub/index.py` — SQLite catalog index with FTS5 search
- `src/rosclaw/hub/client.py` — `FakeRegistryClient`
- `src/rosclaw/hub/auth.py` — profile/token store with JSON fallback
- CLI: `login`, `whoami`, `logout`, `sync`, `search`
- Tests: `tests/hub/test_cache.py`, `tests/hub/test_index.py`,
  `tests/hub/test_cli_hub.py`
- Fake registry fixtures under `tests/fixtures/fake_registry/`

### Phase 3: Verifier, permissions, licenses, policy check CLI

- `src/rosclaw/hub/verifier.py` — manifest, checksum, digest, signature, SBOM/provenance checks
- `src/rosclaw/hub/permissions.py` — dangerous-permission detection and human-approval gates
- `src/rosclaw/hub/licenses.py` — SPDX whitelist, acceptance, data-rights checks
- CLI: `verify`, `policy check`
- Tests: `tests/hub/test_verifier.py`, `tests/hub/test_permissions.py`,
  `tests/hub/test_licenses.py`
- Negative fixtures: tampered checksum, tampered signature, incompatible robot,
  license requiring acceptance

### Phase 4: Installer, lockfile, registry writer, MCP merge, health, rollback

- `src/rosclaw/hub/lifecycle.py` — asset lifecycle state machine
- `src/rosclaw/hub/lockfile.py` — cross-process `AssetsLock` with `filelock`
- `src/rosclaw/hub/resolver.py` — semver, latest, channel, and range resolution
- `src/rosclaw/hub/installer.py` — transactional install with `_rollback()`
- `src/rosclaw/hub/registry_writer.py` — runtime registry JSON files
- `src/rosclaw/hub/mcp_merge.py` — safe `.mcp.json` merge
- `src/rosclaw/hub/health.py` — import, `mcp_list_tools`, and `mcp_call` checks
- CLI: `install`, `uninstall`, `update`, `list`
- Tests: `tests/hub/test_installer_transaction.py`, `tests/hub/test_mcp_merge.py`,
  `tests/hub/test_lockfile.py`

### Phase 5: Publisher, bundle, secret scanning

- `src/rosclaw/hub/publisher.py` — validate, secret-scan, digest, checksums,
  SBOM, provenance, placeholder signing, `.rosclaw` bundle, registry upload
- CLI: `publish` with `--dry-run`, `--private`, `--public`, `--sign`
- Tests: `tests/hub/test_publisher.py`

### Phase 6: E2E tests, security regression, docs, reports

- `tests/hub/test_e2e_fake_registry.py` — full publish → sync → search → install
  → list → uninstall lifecycle
- `tests/hub/test_security_regression.py` — tampering, missing SBOM/provenance,
  dangerous safety config, non-local inbound network, license denial/acceptance,
  secret-scan rejection/warning
- Documentation:
  - `docs/hub/cli.md`
  - `docs/hub/security.md`
  - `docs/hub/publish_guide.md`
  - `docs/hub/private_assets.md`
  - `docs/hub/asset_manifest.md`
- Reports:
  - `docs/evidence/hub/hub_progress.md`
  - `docs/evidence/hub/hub_validation_report.md`
- README / QUICKSTART / `.github/workflows/ci.yml` updated with Hub quickstart
  and a dedicated `hub-test` CI job running `pytest tests/hub -v`

## Test results

```bash
pytest tests/hub -q
# 287 passed
```

## Acceptance commands

```bash
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml
rosclaw hub ref parse rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0
rosclaw hub schema export --format json > /tmp/hub_asset.schema.json

# Use a temporary copy of the fixture registry so publishes do not mutate fixtures.
REGISTRY_DIR=$(mktemp -d)
cp -r tests/fixtures/fake_registry/* "$REGISTRY_DIR/"
python -m tests.fixtures.fake_registry.server --directory "$REGISTRY_DIR" --port 8787 &

export ROSCLAW_HOME=$(mktemp -d)
rosclaw hub login --registry http://localhost:8787 --token fake-valid-token --insecure-local
rosclaw hub whoami
rosclaw hub publish tests/fixtures/hub_assets/skill_valid --private
rosclaw hub sync
rosclaw hub search g1-pick-place
rosclaw hub install rosclaw://skill/rosclaw/g1-pick-place@1.2.0 --yes --skip-health
rosclaw hub list --installed
rosclaw hub uninstall rosclaw://skill/rosclaw/g1-pick-place@1.2.0 --yes
```

## Known limitations / follow-up work

- Signing uses a placeholder HMAC key and dummy certificate. Replace with
  Sigstore / cosign before production.
- Fake registry is file/HTTP based. Cloud registry client is stubbed for later
  implementation.
- `tarfile.extractall()` deprecation warnings are avoided on Python 3.12+ by
  using a compatibility helper with `filter='data'`; older Python versions use
  the legacy extraction path.
- Large model-weight resume download is not implemented.
- Deep `body.yaml` patch integration with `BodyResolver` is partial.

## Conclusion

All Hub phases from the master plan are implemented and validated. The
subsystem is ready for integration testing with the rest of the ROSClaw runtime
and for future cloud registry connectivity.
