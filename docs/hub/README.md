# ROSClaw Hub

ROSClaw Hub is a **Physical-AI Asset Hub** for skills, providers, hardware MCP servers, digital twins, e-URDF profiles, and cognitive wikis.

It is **not** only a Skill Market and **not** only an MCP package manager. It is a unified registry for all assets that an embodied agent needs to operate, validate, and evolve safely.

---

## Asset Types

| Type | Description |
|---|---|
| **Skills** | Reusable embodied task policies, recovery strategies, and skill graphs. |
| **Providers** | LLM, VLM, VLA, VLN, world model, critic, embedding, and classical robotics providers. |
| **Hardware MCP servers** | Agent-facing interfaces for robot bodies, sensors, tools, and lab devices. |
| **Digital twins** | Simulation worlds, robot assets, validation scenes, and replay environments. |
| **e-URDF profiles** | Robot embodiment definitions, safety envelopes, capabilities, and simulation metadata. |
| **Cognitive wikis** | Task cards, failure taxonomies, constraints, evidence, and engineering knowledge. |

---

## Usage Modes

| Mode | Cloud key required | Description |
|---|---|---|
| **Local-only** | No | Use local assets without any cloud connection. Default after `rosclaw firstboot --profile offline`. |
| **Cloud-sync** | Yes | Sync asset catalog and selected metadata with ROSClaw Cloud. |
| **Team-managed** | Yes | Use organization-managed assets, policies, and registries. |

Local-only mode does not require a `ROSCLAW_API_KEY`.

---

## Available CLI Commands

### Search

```bash
rosclaw hub search g1
rosclaw hub search pick_cube --type skill
rosclaw hub search --robot ur5e --compatible
```

### Authentication

```bash
rosclaw hub login --registry https://hub.rosclaw.io --token $ROSCLAW_HUB_TOKEN
rosclaw hub whoami
rosclaw hub logout
```

### Catalog Sync

```bash
rosclaw hub sync
rosclaw hub sync --clear
```

### Validation and Verification

```bash
rosclaw hub validate ./manifest.yaml
rosclaw hub verify ./asset_dir
rosclaw hub policy check ./asset_dir --accept-license
```

### Reference Parsing

```bash
rosclaw hub ref parse rosclaw://hub/rosclaw.io/skill/rosclaw/pick_cube@1.0.0
rosclaw hub schema export --output schema.json
```

---

## Asset Lifecycle

```text
Create
  → Validate
  → Sign
  → Publish
  → Sync
  → Install
  → Activate
  → Evaluate
  → Update / Rollback
  → Uninstall
```

Use `rosclaw-forge` to create assets from SDKs, ROS 2 interfaces, docs, and e-URDF profiles.

---

## Asset URI Scheme

```text
rosclaw://hub/<registry>/<asset-type>/<namespace>/<name>@<version>
```

Example:

```text
rosclaw://hub/rosclaw.io/hardware_mcp/rosclaw/unitree-g1@1.0.0
```

---

## Planned Commands

The following commands are on the roadmap but not yet implemented:

```bash
rosclaw hub install rosclaw://hub/rosclaw.io/skill/rosclaw/pick_cube@1.0.0
rosclaw hub list --installed
```

Until then, install assets manually by downloading and validating them locally.

---

## See Also

- [Physical-AI Assets](../ASSETS.md)
- [CLI Reference](../CLI.md)
- [Forge Workflow](../../README.md)
