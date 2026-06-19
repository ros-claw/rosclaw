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

```bash
# 1. Validate a local asset
rosclaw hub validate tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml

# 2. Start a local fake registry (for testing)
python -m tests.fixtures.fake_registry.server --port 8787

# 3. Login, sync, search
rosclaw hub login --registry http://localhost:8787 --token fake-valid-token --insecure-local
rosclaw hub sync
rosclaw hub search g1

# 4. Install / uninstall
rosclaw hub install rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
rosclaw hub list --installed
rosclaw hub uninstall rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

## Reports

- [`reports/hub_progress.md`](../../reports/hub_progress.md) — implementation summary by phase
- [`reports/hub_validation_report.md`](../../reports/hub_validation_report.md) — acceptance criteria and sign-off

## Known limitations

- Signing uses placeholder material and must be replaced before production.
- The fake registry is local/file-based; cloud registry client is stubbed.
- `tarfile.extractall()` emits deprecation warnings on Python 3.12+.
