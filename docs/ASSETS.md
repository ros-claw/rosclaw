# ROSClaw Generic Physical-AI Asset Hub

The generic `rosclaw hub` subsystem packages and installs typed developer
assets that can extend ROSClaw. It currently provides a verified local
development pipeline and a file/fixture registry for end-to-end testing. It is
not yet the production public catalog shown on the ROSClaw website.

This subsystem is also separate from the older hardware-MCP discovery flow
under `rosclaw mcp`. Their protocols and indexes must not be treated as
interchangeable.

## Asset types

| Type | Purpose |
|------|---------|
| `skill` | Reusable physical-AI task logic |
| `provider` | Runtime capability provider |
| `hardware_mcp` | MCP server integration for hardware |
| `digital_twin` | Simulation model or digital-twin asset |
| `cognitive_wiki` | Structured operational knowledge |

Every asset has a versioned canonical reference:

```text
rosclaw://<type>/<namespace>/<name>@<semver>
```

Example:

```text
rosclaw://skill/rosclaw/g1-pick-place@1.2.0
```

## Implemented lifecycle

```text
manifest.yaml + payload
  -> rosclaw hub validate       schema only
  -> rosclaw hub publish        scan, hash, attest, sign, bundle
  -> local/file registry        test upload and immutable version layout
  -> rosclaw hub sync/search    local catalog cache
  -> rosclaw hub install        verify, authorize, install transaction
  -> rosclaw hub list/update/uninstall
```

The local and remote-reference install paths are implemented. Local source
directories and `.rosclaw` bundles are supported. Both local bundles and
registry bundles use the same bounded safe extractor.

## Manifest

The canonical schema version is `hub.asset.v1`. A manifest contains:

- identity, publisher, visibility, and lifecycle metadata;
- runtime, OS, Python, ROS, CUDA, robot, and hardware compatibility;
- asset and package dependencies;
- hardware, ROS, MCP, filesystem, network, and configuration permissions;
- license and data-rights declarations;
- signing, checksum, SBOM, provenance, and sandbox policy;
- typed artifacts, declarative install entrypoints, registries, and health
  checks;
- type-specific metadata.

The following is a security excerpt, not a complete manifest:

```yaml
schema_version: hub.asset.v1
asset:
  type: skill
  namespace: my-org
  name: guarded-pick
  version: 1.0.0
security:
  signing:
    required: true
    scheme: ed25519
    key_id: my-org-release-2026
    file: signatures/manifest.ed25519
  checksums:
    algorithm: sha256
    file: checksums.txt
  sbom: SBOM.spdx.json
  provenance: PROVENANCE.json
```

See [the manifest reference](hub/asset_manifest.md) for a complete valid
example and field definitions.

## Local verified workflow

The repository includes an intentionally public test key. It is useful for
reproducing the flow, never for release signing:

```bash
export ROSCLAW_HUB_TRUST_STORE="$PWD/tests/fixtures/hub_keys/trust.json"

rosclaw hub validate \
  tests/fixtures/hub_assets/skill_valid/manifest.yaml
rosclaw hub verify tests/fixtures/hub_assets/skill_valid
rosclaw hub install tests/fixtures/hub_assets/skill_valid --dry-run \
  --allow-real-robot
```

Install an already-created bundle without manually calling `tar`:

```bash
rosclaw hub install ./dist/my-org-guarded-pick-1.0.0.rosclaw \
  --trust-store /etc/rosclaw/hub-trust.json \
  --allow-real-robot
```

`--allow-real-robot` permits installation of an asset that declares that
capability. It does not grant runtime action authorization or bypass
`rosclawd`.

## Integrity and publisher trust

The verifier rejects:

- malformed manifests and checksum lines;
- missing, modified, or untracked regular payload files;
- unsafe/control-character paths, symbolic links, and non-regular filesystem entries;
- artifact digest mismatches;
- missing SBOM or provenance files;
- unsupported signature schemes, unknown or inactive keys, scope mismatch,
  and invalid Ed25519 signatures.

Trust roots are supplied independently using `--trust-store` or
`ROSCLAW_HUB_TRUST_STORE`. The packaged trust store is intentionally empty, so
no publisher becomes trusted merely by installing ROSClaw.

## Installation policy

Installation fails closed for these declarations unless their dedicated flags
are present:

- `hardware.real_robot_execution` -> `--allow-real-robot`
- `modifies.safety_config` -> `--allow-safety-config-changes`
- non-local `network.inbound` -> `--allow-network-inbound`

`--yes` only accepts license prompts. It does not grant these permissions.

Hub installation does not execute an asset. Physical execution remains behind
body binding, runtime Permit, Session and Action Lease checks, safety policy,
and a registered daemon-side executor.

## Registry boundary

`FakeRegistryClient` supports local paths, `file://`, and a fixture HTTP
server. It exists for tests and development. There is currently no supported
`https://hub.rosclaw.io` generic registry, governed release-key root, anonymous
production download flow, TUF metadata, or catalog rollback protection.

The website's discovery database stores index and evidence metadata. A website
badge or official publisher label is not a Hub signing trust root and does not
prove simulation, hardware, or Agent verification.

## References

- [Hub documentation](hub/README.md)
- [Manifest reference](hub/asset_manifest.md)
- [Publishing guide](hub/publish_guide.md)
- [Security model](hub/security.md)
- [CLI reference](hub/cli.md)
- [Runtime safety model](SAFETY.md)
