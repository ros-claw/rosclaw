# ROSClaw Hub Asset Manifest

Every Hub asset is described by a single `manifest.yaml` file at the root of the
asset directory. The manifest is validated by `src/rosclaw/hub/schema.py` using
Pydantic v2.

## Top-level schema

```yaml
schema_version: hub.asset.v1
asset:
  type: skill
  namespace: rosclaw
  name: g1-pick-place
  version: 1.2.0
  title: G1 Pick and Place
  summary: Short one-line description
  description: |
    Longer description with behavior, assumptions, and recovery logic.
  tags:
    - g1
    - manipulation
publisher:
  id: rosclaw
  display_name: ROSClaw Team
  trust_level: official
  contact: security@rosclaw.io
visibility:
  scope: public
  allowed_orgs: []
  allowed_users: []
lifecycle:
  status: stable
  channel: stable
  deprecated: false
  yanked: false
  replacement: null
compatibility:
  rosclaw:
    min_version: 1.0.0
    max_version: null
  os: [linux, macos]
  arch: [x86_64, aarch64]
  python:
    requires: '>=3.11,<3.14'
  ros:
    distributions: [humble, jazzy]
    required: false
  cuda:
    required: false
    min_version: null
  robot:
    eurdf_profiles: [unitree/g1]
    body_kinds: [humanoid]
  hardware:
    required_devices: []
  runtime_features: [mcp, sandbox]
dependencies:
  assets: []
  python: []
  system: []
  ros: []
permissions:
  hardware:
    real_robot_execution: true
    actuators: [arms, gripper]
    sensors: [camera, joint_state]
  ros:
    topics_read: []
    topics_write: []
    services: []
    actions: []
  mcp:
    tools: []
  filesystem:
    read: []
    write: []
  network:
    outbound: []
    inbound: []
  modifies:
    mcp_config: false
    body_yaml: false
    rosclaw_yaml: false
    safety_config: false
  requires_human_approval: []
license:
  spdx: MIT
  license_file: LICENSE
  commercial_use: true
  redistribution: true
  attribution_required: true
  export_control: none
data_rights:
  contains_training_data: false
  contains_robot_logs: false
  contains_personal_data: false
  allowed_usage: [research, commercial]
  restrictions: []
security:
  signing:
    required: true
    scheme: ed25519
    key_id: rosclaw-release-2026
    file: signatures/manifest.ed25519
  checksums:
    algorithm: sha256
    file: checksums.txt
  sbom: SBOM.spdx.json
  provenance: PROVENANCE.json
  sandbox_required: true
  network_isolation_recommended: true
artifacts:
  - name: behavior-tree
    kind: xml
    path: artifacts/skill/behavior_tree.xml
    digest: sha256:...
    size_bytes: 4096
install:
  mode: declarative
  entrypoints:
    skill:
      command: python -m rosclaw_skill_g1_pick_place.register
      env: {}
  registries:
    skill: true
    provider: false
    hardware_mcp: false
    digital_twin: false
    cognitive_wiki: false
  health_checks:
    - name: import
      type: python_import
      target: rosclaw_skill_g1_pick_place
special:
  skill:
    task_domain: manipulation
    ...
```

## Asset identity

The canonical reference is:

```text
rosclaw://<type>/<namespace>/<name>@<version>
```

Example: `rosclaw://skill/rosclaw/g1-pick-place@1.2.0`.

### Asset types

- `skill` — executable physical skill
- `provider` — capability provider (perception, planning, etc.)
- `hardware_mcp` — MCP server binding for hardware
- `digital_twin` — simulation model / digital twin asset
- `cognitive_wiki` — documentation / knowledge artifact

## Important sections

### `compatibility`

Defines where the asset can run. The installer checks:

- `rosclaw.min_version` / `max_version`
- `os` and `arch`
- Python version via `python.requires`
- ROS distributions and `required`
- CUDA requirements
- Robot `eurdf_profiles` and `body_kinds`
- Required hardware devices

### `permissions`

Declares everything the asset may do. These are evaluated by
`rosclaw hub policy check` and the installer. Key fields:

- `hardware.real_robot_execution` — requires `--allow-real-robot`
- `ros.topics_write` — motion topics trigger human approval
- `modifies.safety_config` / `modifies.body_yaml` — require
  `--allow-safety-config-changes`
- `network.inbound` — non-local values require `--allow-network-inbound`
- `requires_human_approval` — list of strings such as `real_robot_execution`

### `license` and `data_rights`

License policy uses `spdx`, `license_file`, `commercial_use`,
`redistribution`, `attribution_required`, and `export_control`.

Data rights describe whether the asset contains training data, robot logs, or
personal data, and what usage is allowed.

### `security`

- `signing.required` — whether publication must produce a signature; remote
  installation still requires a trusted signature even if an untrusted
  manifest sets this to `false`
- `signing.scheme` — currently only `ed25519` is accepted for signed bundles
- `signing.key_id` — lookup key in an operator-controlled Hub trust store
- `signing.file` — base64-encoded detached signature beneath `signatures/`;
  the filename must end in `.ed25519`
- `checksums.file` — path to `checksums.txt`
- `sbom` — path to SPDX SBOM
- `provenance` — path to SLSA-style provenance statement
- `sandbox_required` — whether sandbox validation is mandatory
- `network_isolation_recommended` — recommendation for network isolation

The detached signature covers the exact `manifest.yaml` and `checksums.txt`
bytes with a ROSClaw domain separator. Trust keys are not embedded in the
asset. They are resolved from `--trust-store`, `ROSCLAW_HUB_TRUST_STORE`, or
the packaged trust store. The packaged store is intentionally empty until a
public release-key governance process exists.

### `artifacts`

Each artifact has:

- `name` — logical name
- `kind` — file kind (e.g., `xml`, `safetensors`, `python_package`)
- `path` — relative path from asset root
- `digest` — `sha256:<hex>` (computed by publisher)
- `size_bytes` — file size in bytes (computed by publisher)

### `install`

- `mode` — currently `declarative`
- `entrypoints` — per-asset-type launch commands
- `registries` — which runtime registry JSON files should include this asset
- `health_checks` — checks to run after install (`python_import`,
  `mcp_list_tools`, `mcp_call`)

### `special`

Type-specific metadata:

- `skill`: task domain, API schema, required capabilities, supported bodies,
  evaluation benchmark, runtime entrypoint
- `provider`: capability kind, provider class, input/output schemas
- `hardware_mcp`: vendor, model, SDK, transport, e-URDF binding, body.yaml patch
- `digital_twin`: simulator, model files, firewall config
- `cognitive_wiki`: format, indexed sections, source URL

## Validation

Validate a manifest from the command line:

```bash
rosclaw hub validate ./my_asset/manifest.yaml
rosclaw hub validate ./my_asset/manifest.yaml --json
```

`hub validate` checks schema only. Use `rosclaw hub verify` with an independent
trust store to check complete payload hashes and publisher authenticity.

Export the JSON Schema for editors or CI:

```bash
rosclaw hub schema export --format json > hub_asset.schema.json
```

## Example manifests

See the fixture assets:

- `tests/fixtures/hub_assets/skill_valid/manifest.yaml`
- `tests/fixtures/hub_assets/hardware_mcp_valid/manifest.yaml`
- `tests/fixtures/hub_assets/provider_valid/manifest.yaml`
- `tests/fixtures/hub_assets/digital_twin_valid/manifest.yaml`
- `tests/fixtures/hub_assets/cognitive_wiki_valid/manifest.yaml`
- `tests/fixtures/hub_assets/license_requires_acceptance/manifest.yaml`

## See also

- [CLI reference](cli.md)
- [Publish guide](publish_guide.md)
- [Security model](security.md)
- `src/rosclaw/hub/schema.py`
- `tests/hub/test_schema.py`
