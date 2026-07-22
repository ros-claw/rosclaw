# ROSClaw Hub CLI Reference

The `rosclaw hub` command group manages ROSClaw Hub assets: skills, providers,
hardware MCP servers, digital twins, and cognitive wikis.

```text
rosclaw hub [-h]
            {validate,ref,schema,login,whoami,logout,sync,search,verify,policy,
             install,uninstall,update,list,publish}
            ...
```

## Command Summary

| Group | Command | Purpose |
|-------|---------|---------|
| Development | `validate` | Validate a `manifest.yaml` against the Hub schema |
| | `ref parse` | Parse a `rosclaw://` URI |
| | `schema export` | Export the manifest JSON Schema |
| Registry auth | `login` | Store registry credentials |
| | `whoami` | Show the active registry profile |
| | `logout` | Remove stored credentials |
| Catalog | `sync` | Download the catalog and build the local SQLite index |
| | `search` | Search the local catalog index |
| Security | `verify` | Verify checksums, artifact digests, signatures, SBOM/provenance |
| | `policy check` | Check permission and license policy locally |
| Lifecycle | `install` | Install an asset from a local directory or registry reference |
| | `uninstall` | Remove an installed asset |
| | `update` | Replace an installed asset with a new local version |
| | `list` | List installed assets |
| Publish | `publish` | Prepare, sign, bundle, and upload an asset |

## Global help

```bash
rosclaw hub --help
```

## Development commands

### `rosclaw hub validate`

Validate a local `manifest.yaml` against the Pydantic v2 manifest schema.

```text
rosclaw hub validate [-h] [--json] manifest
```

| Argument | Description |
|----------|-------------|
| `manifest` | Path to `manifest.yaml` |
| `--json` | Output validation result as JSON |

Example:

```bash
rosclaw hub validate tests/fixtures/hub_assets/skill_valid/manifest.yaml
rosclaw hub validate tests/fixtures/hub_assets/skill_valid/manifest.yaml --json
```

JSON output example:

```json
{
  "valid": true,
  "asset": {
    "type": "skill",
    "namespace": "rosclaw",
    "name": "g1-pick-place",
    "version": "1.2.0",
    "title": "G1 Pick and Place"
  }
}
```

### `rosclaw hub ref parse`

Parse a `rosclaw://` asset URI into its components.

```text
rosclaw hub ref parse [-h] [--json] ref
```

Example:

```bash
rosclaw hub ref parse rosclaw://skill/rosclaw/g1-pick-place@1.2.0 --json
```

Output:

```json
{
  "type": "skill",
  "namespace": "rosclaw",
  "name": "g1-pick-place",
  "version": "1.2.0",
  "canonical": "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"
}
```

Reference format:

```text
rosclaw://<type>/<namespace>/<name>@<version_or_range>
```

Supported types: `skill`, `provider`, `hardware_mcp`, `digital_twin`,
`cognitive_wiki`.

### `rosclaw hub schema export`

Export the manifest JSON Schema for use in editors, CI, or generated forms.

```text
rosclaw hub schema export [-h] [--format {json,yaml}] [--output OUTPUT]
```

Example:

```bash
rosclaw hub schema export --format json > /tmp/hub_asset.schema.json
rosclaw hub schema export --format yaml --output docs/hub/asset.schema.yaml
```

## Registry authentication

### `rosclaw hub login`

Store a registry profile and token. The active profile is used by `sync`,
`search`, and `install rosclaw://...`.

```text
rosclaw hub login [-h] --registry REGISTRY --token TOKEN [--insecure-local]
```

| Option | Description |
|--------|-------------|
| `--registry` | Registry base URL |
| `--token` | Access token |
| `--insecure-local` | Allow plain HTTP / local file registries (testing only) |

Example:

```bash
rosclaw hub login \
  --registry http://localhost:8787 \
  --token fake-valid-token \
  --insecure-local
```

Credentials are stored atomically in `~/.rosclaw/config/hub_auth.json`. On
POSIX systems the `config` directory is forced to mode `0700` and the file to
mode `0600`; symbolic links, non-regular files, and files owned by another user
are rejected. The store is still plaintext JSON, so production deployments may
prefer an OS keyring or external secret manager.

### `rosclaw hub whoami`

Show the active registry profile.

```text
rosclaw hub whoami [-h]
```

### `rosclaw hub logout`

Remove credentials for the active registry or a specific registry.

```text
rosclaw hub logout [-h] [--registry REGISTRY]
```

## Catalog commands

### `rosclaw hub sync`

Download `catalog.jsonl` from a registry and index it in the local SQLite
catalog under `~/.rosclaw/hub/indexes/<registry>/catalog.sqlite`.

```text
rosclaw hub sync [-h] [--registry REGISTRY] [--clear]
```

| Option | Description |
|--------|-------------|
| `--registry` | Registry URL (defaults to active profile) |
| `--clear` | Delete the existing index before syncing |

Example:

```bash
rosclaw hub sync
rosclaw hub sync --registry http://localhost:8787 --clear
```

### `rosclaw hub search`

Search the local catalog index. The index is built by `sync`.

```text
rosclaw hub search [-h] [--type TYPE] [--namespace NAMESPACE] [--official]
                   [--license LICENSE] [--robot ROBOT] [--compatible]
                   [--limit LIMIT] [--json]
                   [query]
```

| Option | Description |
|--------|-------------|
| `query` | Free-text search keywords |
| `--type` | Filter by asset type |
| `--namespace` | Filter by publisher namespace |
| `--official` | Only official publishers |
| `--license` | Filter by SPDX identifier (repeatable) |
| `--robot` | Filter by robot profile or body kind |
| `--compatible` | Only assets compatible with this machine |
| `--limit` | Maximum number of results |
| `--json` | Output as JSON |

Example:

```bash
rosclaw hub search pick-place
rosclaw hub search g1 --type skill --official --limit 5
```

## Security commands

### `rosclaw hub verify`

Verify a local asset directory without installing it. Checks:

- Manifest schema validation
- strict `checksums.txt` parsing and complete regular-payload coverage
- Declared artifact digests match on-disk files
- trusted, scope-matched detached Ed25519 signature validity
- path containment and symbolic-link rejection
- SBOM / provenance file existence (when declared)

```text
rosclaw hub verify [-h] [--no-signature] [--trust-store TRUST_STORE]
                   [--json] asset_dir
```

| Option | Description |
|--------|-------------|
| `asset_dir` | Path to the asset directory |
| `--no-signature` | Skip trusted signature verification; local development only |
| `--trust-store` | Independent Hub trusted-key JSON; otherwise use `ROSCLAW_HUB_TRUST_STORE` or packaged keys |
| `--json` | Output as JSON |

Example:

```bash
rosclaw hub verify tests/fixtures/hub_assets/skill_valid \
  --trust-store tests/fixtures/hub_keys/trust.json
rosclaw hub verify tests/fixtures/hub_assets/skill_valid \
  --trust-store tests/fixtures/hub_keys/trust.json --json
```

### `rosclaw hub policy check`

Evaluate permission and license policy for an asset directory.

```text
rosclaw hub policy check [-h] [--allow-real-robot] [--accept-license] [--json]
                         asset_dir
```

| Option | Description |
|--------|-------------|
| `asset_dir` | Path to the asset directory |
| `--allow-real-robot` | Allow assets that request real robot execution |
| `--accept-license` | Explicitly accept licenses that require manual acceptance |
| `--json` | Output as JSON |

Example:

```bash
rosclaw hub policy check tests/fixtures/hub_assets/skill_valid --json
```

Output example:

```json
{
  "ok": true,
  "permissions": {
    "allowed": true,
    "requires_human_approval": true,
    "dangerous_permissions": [
      "hardware.real_robot_execution",
      "ros.topics_write:{'/cmd_vel'}",
      "requires_human_approval:real_robot_execution"
    ],
    "issues": []
  },
  "license": {
    "accepted": true,
    "requires_acceptance": false,
    "issues": []
  }
}
```

## Lifecycle commands

### `rosclaw hub install`

Install an asset from a local directory or a `rosclaw://` registry reference.
The install transaction verifies the asset, checks policy, resolves
dependencies, then acquires the cross-process `assets.lock`, copies files,
updates registries, optionally merges MCP config, runs health checks, and
records the result. A remote dry-run still downloads, safely extracts, binds,
and verifies the bundle.

```text
rosclaw hub install [-h] [--dry-run] [--yes] [--accept-license]
                    [--no-mcp-merge] [--skip-health] [--no-verify-signature]
                    [--trust-store TRUST_STORE]
                    [--allow-real-robot] [--allow-safety-config-changes]
                    [--allow-network-inbound] [--json]
                    asset_dir
```

| Option | Description |
|--------|-------------|
| `asset_dir` | Local directory, `.rosclaw` bundle, or `rosclaw://` reference |
| `--dry-run` | Simulate without writing files |
| `--yes` | Accept license prompts; does not grant safety permissions |
| `--accept-license` | Explicitly accept the asset license |
| `--no-mcp-merge` | Skip updating `.mcp.json` / MCP config |
| `--skip-health` | Skip post-install health checks |
| `--no-verify-signature` | Skip trusted signature verification; local development only |
| `--trust-store` | Independent Hub trusted-key JSON |
| `--allow-real-robot` | Allow real robot execution |
| `--allow-safety-config-changes` | Allow modifications to safety configuration |
| `--allow-network-inbound` | Allow non-local inbound network access |
| `--json` | Output as JSON |

Local install example:

```bash
rosclaw hub install tests/fixtures/hub_assets/hardware_mcp_valid --dry-run \
  --trust-store tests/fixtures/hub_keys/trust.json --allow-real-robot
rosclaw hub install tests/fixtures/hub_assets/hardware_mcp_valid --yes \
  --trust-store tests/fixtures/hub_keys/trust.json --allow-real-robot
```

Registry install example:

```bash
rosclaw hub sync
rosclaw hub install rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

### `rosclaw hub uninstall`

Remove an installed asset and clean up its registry entry, MCP config, lockfile
record, and installed-state file.

```text
rosclaw hub uninstall [-h] [--yes] [--json] ref
```

Example:

```bash
rosclaw hub uninstall rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0 --yes
```

### `rosclaw hub update`

Replace an installed asset with a new version from a local directory.

```text
rosclaw hub update [-h] [--dry-run] [--yes] [--accept-license]
                   [--no-mcp-merge] [--skip-health] [--no-verify-signature]
                   [--trust-store TRUST_STORE]
                   [--allow-real-robot] [--allow-safety-config-changes]
                   [--allow-network-inbound] [--json]
                   ref asset_dir
```

Example:

```bash
rosclaw hub update rosclaw://skill/rosclaw/g1-pick-place@1.2.0 ./g1-pick-place-v1.3.0 --yes
```

### `rosclaw hub list`

List assets recorded in the local lockfile. With `--installed`, only show
installed assets.

```text
rosclaw hub list [-h] [--installed] [--json]
```

Example:

```bash
rosclaw hub list --installed
rosclaw hub list --installed --json
```

## Publish command

### `rosclaw hub publish`

Prepare and publish an asset. `prepare` validates the manifest, scans for
secrets, computes artifact digests, generates `checksums.txt`, SBOM, and
provenance, optionally signs the bundle, and creates a `.rosclaw` tar.gz
bundle. When a registry is configured, the bundle is uploaded.

```text
rosclaw hub publish [-h] [--dry-run] [--private] [--public] [--sign]
                    [--signing-key SIGNING_KEY]
                    [--signing-key-id SIGNING_KEY_ID]
                    [--registry REGISTRY] [--output OUTPUT] [--json]
                    asset_dir
```

| Option | Description |
|--------|-------------|
| `asset_dir` | Source asset directory containing `manifest.yaml` |
| `--dry-run` | Prepare and scan without creating an output bundle or registry entry |
| `--private` | Publish as a private asset |
| `--public` | Publish as a public asset |
| `--sign` | Create a detached Ed25519 signature |
| `--signing-key` | Ed25519 PKCS8 PEM private key outside the asset directory |
| `--signing-key-id` | Trust-store key ID declared by the manifest |
| `--registry` | Registry URL (defaults to active profile) |
| `--output` | Write the bundle to this path or directory |
| `--json` | Output as JSON |

`--output` is a complete local destination and does not require registry
login. Without `--output`, `--registry`, or an active registry profile, a
non-dry-run publish fails instead of choosing an implicit destination.

Example:

```bash
rosclaw hub publish ./my_skill --dry-run --sign \
  --signing-key /secure/path/hub-signing-key.pem \
  --signing-key-id my-org-hub-release-2026
rosclaw hub publish ./my_skill --private --sign \
  --signing-key /secure/path/hub-signing-key.pem \
  --signing-key-id my-org-hub-release-2026 \
  --registry file:///tmp/rosclaw-registry
```

See [publish_guide.md](publish_guide.md) for the full publishing workflow.

## Exit codes

Hub commands use the standard `rosclaw` CLI exit code conventions:

- `0` — success
- non-zero — failure, with a human-readable message and an optional
  `suggested_fix` in JSON output

Common failure reasons include manifest validation errors (`MANIFEST_INVALID`),
checksum/signature verification failures (`CHECKSUM_MISMATCH`), registry
identity or archive failures (`INDEX_VERIFY_FAILED`), permission denials
(`PERMISSION_DENIED`), license denials (`LICENSE_DENIED`), missing assets
(`ASSET_NOT_FOUND`), and authentication failures (`AUTH_REQUIRED`).
