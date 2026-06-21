# ROSClaw Hub Validation Report

**Date:** 2026-06-22
**Scope:** `src/rosclaw/hub/`, `tests/hub/`, Hub documentation, README/QUICKSTART/CI updates
**Objective:** Confirm that the Hub subsystem meets the acceptance criteria from
the master implementation plan.

## Validation criteria

| # | Criterion | Method | Result |
|---|-----------|--------|--------|
| 1 | Code passes lint | `ruff check src/rosclaw/hub tests/hub` | PASS |
| 2 | Code is formatted | `ruff format --check src/rosclaw/hub tests/hub` | PASS |
| 3 | Unit + integration tests pass | `pytest tests/hub -q` | PASS (287 passed) |
| 4 | Type check on Hub code | `mypy src/rosclaw/hub` | PASS |
| 5 | E2E lifecycle works | `tests/hub/test_e2e_fake_registry.py` | PASS |
| 6 | Security regressions covered | `tests/hub/test_security_regression.py` | PASS |
| 7 | Client and publisher coverage | `tests/hub/test_client.py`, expanded `tests/hub/test_publisher.py` | PASS |
| 8 | Documentation written | `docs/hub/*.md` created | PASS |
| 9 | Progress / validation reports written | `reports/hub_*.md` created | PASS |
| 10 | README / QUICKSTART updated | Hub quickstart added | PASS |
| 11 | CI updated | Dedicated `hub-test` job added to `.github/workflows/ci.yml` | PASS |
| 12 | Installer transaction / rollback tests | `tests/hub/test_installer_transaction.py` | PASS |
| 13 | MCP config merge tests | `tests/hub/test_mcp_merge.py` | PASS |

## Commands run

```bash
ruff check src/rosclaw/hub tests/hub
ruff format --check src/rosclaw/hub tests/hub
pytest tests/hub -q
mypy src/rosclaw/hub
```

`mypy src/rosclaw/hub` is now clean thanks to Hub-scoped overrides in
`pyproject.toml`: Hub modules are checked with normal imports while the rest of
`rosclaw.*` is followed with `skip` to avoid pre-existing type errors in
unrelated modules.

CI workflow additionally runs `pytest tests/hub -v` in a dedicated `hub-test` job
after `lint` and `type-check` succeed.

## Pull request

- **PR:** [#18 feat(hub): Phase 6 completion — installer transaction/rollback tests, MCP merge tests, and lockfile cleanup fix](https://github.com/ros-claw/rosclaw/pull/18)
- **Branch:** `hub/phase6-completion` → `main`
- **Status:** OPEN; CI checks in progress/queued at time of report

## Detailed findings

### Lint and format

- `ruff check` reports no errors across `src/rosclaw/hub` and `tests/hub`.
- `ruff format --check` reports all 36 files already formatted.

### Tests

- `pytest tests/hub -q` reports **287 passed**.
- Test modules cover schema, refs, cache, index, CLI, verifier, permissions,
  licenses, lockfile, installer transaction, MCP merge, publisher, registry
  client, E2E lifecycle, and security regression.

### Transaction and rollback

`tests/hub/test_installer_transaction.py` validates:

1. Happy-path local install writes the target directory, lockfile entry,
   cache record, registry entry, and managed MCP server.
2. Registry failure after file copy rolls back the target directory, lockfile,
   cache record, and MCP state.
3. Post-MCP failure rolls back the registry entry, MCP fragment, lockfile,
   and cache record.
4. Re-installing an already-installed asset raises `ASSET_ALREADY_INSTALLED`.
5. Uninstall removes the target directory, lockfile entry, cache record,
   registry entry, and MCP server entry.

A real bug was found and fixed during this work: `_rollback()` in
`src/rosclaw/hub/installer.py` now removes the lockfile entry and saves the
lockfile, preventing partial-install state leaks.

### Client

`tests/hub/test_client.py` (new) covers both local and HTTP branches of
`FakeRegistryClient`:

1. Registry URL normalization: trailing-slash stripping, `file://` conversion,
   plain local paths.
2. `sync()` parses valid JSONL catalog lines and raises `INDEX_VERIFY_FAILED`
   for malformed lines.
3. `fetch_manifest()` and `fetch_bundle()` require a versioned `AssetRef` and
   return YAML/bytes for valid local references.
4. `fetch_blob()` validates digest format (`sha256:<hex>`).
5. `whoami()` returns the fake profile for `fake-valid-token` and raises
   `AUTH_FAILED` otherwise.
6. `publish_bundle()` writes manifests and catalog entries for local registries.
7. HTTP `publish_bundle()` POSTs bytes, parses JSON responses, falls back to
   the upload URL for empty bodies, and maps HTTP 401/409/500 plus `URLError`
   to the correct `HubErrorCode`.
8. HTTP GET error handling maps 404 to `ASSET_NOT_FOUND` and other errors to
   `REGISTRY_UNREACHABLE`.

### Publisher

`tests/hub/test_publisher.py` is expanded with edge-case coverage for previously
uncovered branches:

1. `scan_secrets()` skips binary files.
2. `prepare()` warns when a declared artifact is missing on disk.
3. `publish()` raises `ASSET_NOT_FOUND` for non-existent asset directories.
4. `options.visibility` overrides the manifest visibility scope.
5. `options.sign=True` forces signing even when the manifest does not require it.
6. `bundle()` writes into an existing output directory when `output` is a
   directory.
7. `publish()` builds a registry client from `options.registry` via the auth
   store when no explicit client is passed.
8. Registry upload failures that are `HubError` instances are re-raised with
   their original code.
9. Unexpected registry upload exceptions are wrapped as
   `REGISTRY_UNREACHABLE`.

### MCP config merge

`tests/hub/test_mcp_merge.py` validates:

1. `add_server()` writes `.mcp.json` and a per-server runtime fragment.
2. Re-adding the same asset is idempotent (overwrites, never duplicates).
3. `remove_server()` deletes the entry and fragment but preserves unmanaged
   servers.
4. `list_servers()` only returns rosclaw-managed entries.
5. `is_managed()` reflects whether an asset's server is present.
6. Corrupt `.mcp.json` and missing-entrypoint-command errors raise `HubError`.

### End-to-end lifecycle

`test_full_lifecycle_publish_sync_search_install_list_uninstall` exercises:

1. Publishing a valid skill to the fake registry.
2. Verifying registry files (`manifests/...`, `bundles/...`, `catalog.jsonl`).
3. Syncing the catalog into SQLite and caching manifests.
4. Searching the catalog.
5. Installing by `rosclaw://` reference.
6. Checking the lockfile, runtime registry, and installed-state JSON.
7. Listing installed assets.
8. Uninstalling and verifying removal.

`test_install_by_ref_requires_cached_manifest` confirms that installing by
reference fails gracefully when the manifest has not been cached by `sync`.

### Security regression coverage

`tests/hub/test_security_regression.py` includes:

- Tampered `checksums.txt` detected by verifier and installer.
- Tampered artifact digest detected by verifier and installer.
- Missing SBOM / provenance files detected.
- Dangerous `safety_config` modification blocked without
  `--allow-safety-config-changes`.
- Non-local inbound network access blocked without
  `--allow-network-inbound`.
- License denial without `--accept-license`.
- License acceptance with `--accept-license`.
- Secret-scan publish rejection in strict mode.
- Secret-scan warning in non-failing mode.

### Documentation

- `docs/hub/cli.md` — command reference for all `rosclaw hub` subcommands.
- `docs/hub/security.md` — threat model, verification, permission/license
  policy, install-time guards.
- `docs/hub/publish_guide.md` — full publishing workflow.
- `docs/hub/private_assets.md` — private/internal asset handling.
- `docs/hub/asset_manifest.md` — manifest schema reference.

### Reports

- `reports/hub_progress.md` — implementation summary by phase.
- `reports/hub_validation_report.md` — this report.

### README / QUICKSTART / CI

- `README.md` updated with a Hub quickstart section, feature summary, and
  repository structure note.
- `QUICKSTART.md` updated with Hub quickstart steps.
- `.github/workflows/ci.yml` updated with a dedicated `hub-test` job that runs
  `pytest tests/hub -v` after `lint` and `type-check` succeed.

## Known issues

- Placeholder signing material is present and must be replaced before
  production use.
- Coverage gaps in `src/rosclaw/hub/client.py` and `src/rosclaw/hub/publisher.py`
  have been filled by `tests/hub/test_client.py` and the expanded
  `tests/hub/test_publisher.py`.

## Sign-off

The ROSClaw Hub subsystem satisfies the Phase 6 acceptance criteria and is
ready for wider runtime integration and future cloud-registry work.

**Validated by:** Claude Code / automated test suite
**Date:** 2026-06-22
