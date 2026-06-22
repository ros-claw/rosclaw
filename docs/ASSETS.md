# ROSClaw Physical-AI Assets

## What is the Physical-AI Asset Hub?

ROSClaw Hub is a **Physical-AI Asset Hub** for skills, providers, hardware MCP servers, digital twins, e-URDF profiles, and cognitive wikis. It supports local-only usage, cloud-synced registries, and team-managed private registries.

The Hub is **not** only a Skill Market and **not** only an MCP package manager. It is a unified registry for all assets that an embodied agent needs to operate, validate, and evolve safely.

---

## Asset Types

| Type | Description |
|---|---|
| `skill` | Task policy, recovery logic, skill graph, parameters, evaluation metadata |
| `provider` | LLM / VLM / VLA / VLN / world model / critic / embedding / robotics algorithm |
| `hardware_mcp` | MCP-compatible robot, sensor, or device interface |
| `digital_twin` | Simulation world, robot asset, replay scene, validation environment |
| `e_urdf` | Robot embodiment profile and safety envelope |
| `cognitive_wiki` | TaskCard, failure taxonomy, constraints, evidence, and repair knowledge |

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

- **Create** — author the asset or generate it with `rosclaw-forge`.
- **Validate** — run `rosclaw hub validate` against the manifest schema.
- **Sign** — optional cryptographic signature for provenance.
- **Publish** — upload to a registry.
- **Sync** — download the catalog index with `rosclaw hub sync`.
- **Install** — stage the asset in `~/.rosclaw/hub/`.
- **Activate** — register the asset with the runtime.
- **Evaluate** — run benchmarks and sandbox tests.
- **Update / Rollback** — versioned replacement or revert.
- **Uninstall** — remove the asset from the workspace.

---

## Local-only Assets

Local assets can be created, validated, and used without a ROSClaw Cloud key:

```bash
rosclaw hub validate ./my_skill/manifest.yaml
rosclaw hub verify ./my_skill/
```

Local-only mode is the default after `rosclaw firstboot --profile offline`.

---

## Cloud-synced Assets

Cloud sync, private team assets, and organization-managed registries may require authentication:

```bash
rosclaw hub login --registry https://hub.rosclaw.io --token $ROSCLAW_HUB_TOKEN
rosclaw hub sync
rosclaw hub search pick_cube --type skill
```

Cloud sync is opt-in. Telemetry and artifact upload are disabled by default.

---

## Asset Reference URI

ROSClaw uses a `rosclaw://` URI scheme for assets:

```text
rosclaw://hub/<registry>/<asset-type>/<namespace>/<name>@<version>
```

Example:

```text
rosclaw://hub/rosclaw.io/hardware_mcp/rosclaw/unitree-g1@1.0.0
```

---

## Forge Workflow

`rosclaw-forge` compiles SDKs, ROS 2 interfaces, docs, and e-URDF profiles into distributable assets:

```text
SDK / ROS 2 Interfaces / Docs / e-URDF
        ↓
  rosclaw-forge
        ↓
MCP Server + Skill Manifest + Provider Manifest + Tests + Hub Metadata
```

Example:

```bash
rosclaw forge sdk-to-mcp \
  --name unitree_go2 \
  --sdk-docs ./docs/unitree_go2_sdk.md \
  --output ./generated/unitree_go2_bundle

rosclaw forge validate ./generated/unitree_go2_bundle
rosclaw forge install ./generated/unitree_go2_bundle --staging
```

---

## Hub CLI

Currently available Hub commands:

```bash
rosclaw hub search g1
rosclaw hub validate ./manifest.yaml
rosclaw hub verify ./asset_dir
rosclaw hub policy check ./asset_dir
rosclaw hub sync
rosclaw hub login --registry https://hub.rosclaw.io --token $TOKEN
rosclaw hub whoami
rosclaw hub logout
```

Planned commands:

```bash
rosclaw hub install rosclaw://hub/rosclaw.io/skill/rosclaw/pick_cube@1.0.0
rosclaw hub list --installed
```

---

## Team-managed Registries

Organizations can run private registries with custom policies:

- Require signed manifests
- Restrict real-robot assets
- Enforce license acceptance
- Gate skill promotion by team role

Use `rosclaw hub policy check` to validate an asset against local policy before activation.
