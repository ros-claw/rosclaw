# ROSClaw Physical-AI Asset Hub

The ROSClaw Hub is a **Physical-AI Asset Hub**: a registry-aware system for discovering, validating, distributing, and versioning assets that connect AI agents to the physical world.

---

## What is a Physical-AI Asset?

A Physical-AI asset is any versioned, self-describing package that:

- Declares what physical capabilities it requires.
- Declares what physical capabilities it provides.
- Carries a manifest with identity, provenance, license, and verification policy.
- Can be validated against the ROSClaw manifest schema.
- Can be installed into a local workspace or synced from a registry.

Assets are not generic files. They are typed, validated, and runtime-aware.

---

## Asset Types

| Type | Description | Example |
|------|-------------|---------|
| `skill` | Reusable physical-AI skill | `pick_cube`, `open_door` |
| `provider` | Runtime capability provider | vision provider, LLM router |
| `hardware_mcp` | MCP server that wraps real hardware | Unitree G1 MCP, UR5e MCP |
| `digital_twin` | Simulation asset / e-URDF twin | MuJoCo scene, Isaac Sim asset |
| `e_urdf` | Extended URDF model with capabilities | `unitree-g1@1.0.0` |
| `cognitive_wiki` | Structured operational knowledge | maintenance procedures, safety notes |

---

## Asset Lifecycle

```
Author
    ↓
Write manifest.yaml + payload
    ↓
rosclaw hub validate   (local schema check)
    ↓
rosclaw hub publish --dry-run   (bundle + sign simulation)
    ↓
Upload to registry
    ↓
Registry verification + policy check
    ↓
rosclaw hub sync   (consumer fetches index)
    ↓
rosclaw hub search / info   (consumer discovers)
    ↓
rosclaw hub install   (consumer installs locally)  [Planned]
    ↓
Runtime loads asset and checks body compatibility
    ↓
Sandbox validates every use
    ↓
Practice records evidence
    ↓
Auto / Darwin may propose updates
```

---

## Asset Identity

Assets are identified by URIs of the form:

```text
rosclaw://{type}/{namespace}/{name}@{version}
```

Examples:

```text
rosclaw://skill/rosclaw/pick_cube@1.2.0
rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0
rosclaw://digital_twin/rosclaw/tabletop@0.4.1
```

Identity rules:

- `type` must be one of the supported asset types.
- `namespace` identifies the publisher or organization.
- `name` is kebab-case and unique within `{type}/{namespace}`.
- `version` follows [SemVer](https://semver.org/).

---

## Manifest Schema

Every asset contains a `manifest.yaml` with at least:

```yaml
apiVersion: hub.rosclaw.io/v1
kind: Asset
metadata:
  id: rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0
  name: unitree-g1
  namespace: rosclaw
  version: 1.0.0
spec:
  type: hardware_mcp
  description: MCP server for Unitree G1 humanoid
  requires:
    capabilities:
      - name: ros2-control
      - name: humanoid-locomotion
  provides:
    capabilities:
      - name: walk
      - name: stand
  license: MIT
  verification:
    signed: false
    checksums:
      sha256: abc123...
```

See [docs/hub/asset_manifest.md](hub/asset_manifest.md) for the full schema.

---

## Local-Only Assets

You can use assets without any registry:

```bash
# Validate a local manifest
rosclaw hub validate ./my_skill/manifest.yaml

# Use it in a sandbox run
rosclaw sandbox run --skill ./my_skill --robot sim_ur5e
```

Local-only assets are useful for:

- Air-gapped robots.
- Proprietary skills that must not leave the lab.
- Development and debugging.

---

## Cloud-Synced Assets

To discover and sync assets from a registry:

```bash
rosclaw hub login --registry https://hub.rosclaw.io --token $ROSCLAW_HUB_TOKEN
rosclaw hub sync
rosclaw hub search pick
rosclaw hub info rosclaw://skill/rosclaw/pick_cube@1.2.0
rosclaw hub install rosclaw://skill/rosclaw/pick_cube@1.2.0
rosclaw hub list --installed
rosclaw hub uninstall rosclaw://skill/rosclaw/pick_cube@1.2.0
```

---

## Verification and Trust

Before an asset is loaded by the runtime:

1. The manifest schema is validated.
2. Checksums are verified against the payload.
3. The license and permission policy are checked.
4. Required capabilities are matched against the effective body model.
5. The Sandbox validates every skill execution.

Currently, signing uses placeholder material and must be replaced before production. See [docs/hub/security.md](hub/security.md).

---

## CLI Reference

See [docs/CLI.md](CLI.md) for the full list of `rosclaw hub` commands and their status.

---

## See Also

- [docs/hub/README.md](hub/README.md) — Hub workflows and registry setup.
- [docs/hub/asset_manifest.md](hub/asset_manifest.md) — Full manifest schema.
- [docs/SAFETY.md](SAFETY.md) — Safety model.
- [QUICKSTART.md](../QUICKSTART.md) — Quick start.
