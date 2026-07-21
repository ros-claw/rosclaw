# ROSClaw Hub Documentation

The ROSClaw Hub turns the local runtime into an asset-aware platform for
physical-AI skills, providers, hardware MCP servers, digital twins, and
cognitive wikis.

## Quick links

| Document | What it covers |
|----------|----------------|
| [Asset Manifest Reference](asset_manifest.md) | Full YAML schema, identity rules, and validation examples |
| [CLI Reference](cli.md) | Every `rosclaw hub` subcommand, options, JSON output, and exit codes |
| [Publishing Guide](publish_guide.md) | Dry-run → signed bundle → registry upload workflow |
| [Security Model](security.md) | Threat model, verification, permission/license policy, and audit trail |
| [Private Assets](private_assets.md) | Visibility scopes, offline install, and local registry testing |

## Supported asset types

- `skill` — reusable physical-AI skill
- `provider` — runtime capability provider
- `hardware_mcp` — MCP server that wraps real hardware
- `digital_twin` — simulation asset / e-URDF twin
- `cognitive_wiki` — structured operational knowledge

## One-page workflow

The repository fixture key is intentionally public and is suitable only for
this local demonstration. Use a temporary registry so the workflow does not
mutate committed fixtures:

```bash
# 1. Validate a local asset
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml

# 2. Configure test-only signing and trust material
export ROSCLAW_HUB_SIGNING_KEY="$PWD/tests/fixtures/hub_keys/fixture-private.pem"
export ROSCLAW_HUB_SIGNING_KEY_ID=rosclaw-hub-fixture-v1
export ROSCLAW_HUB_TRUST_STORE="$PWD/tests/fixtures/hub_keys/trust.json"

# 3. Start an empty local fake registry
REGISTRY_DIR=$(mktemp -d)
touch "$REGISTRY_DIR/catalog.jsonl"
PYTHONPATH=src python3 tests/fixtures/fake_registry/server.py \
  --directory "$REGISTRY_DIR" --port 8787 &

# 4. Login, publish a signed skill, sync, and search
export ROSCLAW_HOME=$(mktemp -d)
rosclaw hub login --registry http://localhost:8787 --token fake-valid-token --insecure-local
rosclaw hub publish tests/fixtures/hub_assets/skill_valid --private --sign
rosclaw hub sync
rosclaw hub search g1-pick-place

# 5. Verify, install, and uninstall the published asset
rosclaw hub verify tests/fixtures/hub_assets/skill_valid
rosclaw hub install rosclaw://skill/rosclaw/g1-pick-place@1.2.0 \
  --yes --allow-real-robot --skip-health
rosclaw hub list --installed
rosclaw hub uninstall rosclaw://skill/rosclaw/g1-pick-place@1.2.0 --yes
```

## Reports

- [`reports/hub_progress.md`](../../reports/hub_progress.md) — implementation summary by phase
- [`reports/hub_validation_report.md`](../../reports/hub_validation_report.md) — acceptance criteria and sign-off

## Known limitations

- Detached Ed25519 signing, scoped trust, complete payload integrity, and safe
  extraction are implemented and tested on Python 3.11 through 3.13.
- The packaged trust store is intentionally empty; there is no governed public
  generic-Hub release root yet.
- The registry is local/file-backed. A production public client, TUF metadata,
  rollback protection, and builder-attestation verification remain open.
- The generic `rosclaw hub` registry is separate from the existing
  `rosclaw mcp` discovery service.
