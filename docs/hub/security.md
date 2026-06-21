# ROSClaw Hub Security Model

The ROSClaw Hub is a distribution system for executable physical-AI assets. A
single malicious or accidentally dangerous asset could move a real robot, alter
safety limits, or exfiltrate logs. The Hub therefore uses **defense in depth**:
manifest schema validation, integrity verification, permission policy, license
policy, secret scanning, sandbox gating, and rollback on failure.

## Threat model

| Threat | Mitigation |
|--------|------------|
| Tampered asset after download | sha256 checksums on every file |
| Man-in-the-middle registry | Placeholder signing + certificate checks (Sigstore-ready) |
| Leaked secrets in published asset | Secret scan at publish time |
| Unauthorized real-robot motion | Permission policy requires explicit opt-in |
| Safety configuration changes | `safety_config` and `body_yaml` flags require approval |
| Incompatible robot / OS / ROS | Compatibility checks before install |
| Network exfiltration | Network policy restricts inbound/outbound scope |
| License violation | SPDX whitelist, acceptance gates, data-rights checks |
| Failed install leaves system dirty | Transactional install with `_rollback()` cleanup |
| Malicious registry upload | Registry token auth; future TUF-style metadata planned |

## Verification (`rosclaw hub verify`)

The verifier (`src/rosclaw/hub/verifier.py`) checks an asset directory without
executing anything:

1. `manifest.yaml` loads and validates against the Pydantic schema.
2. The checksums file exists and matches the security section.
3. Every file listed in `checksums.txt` exists and its sha256 digest matches.
4. Every artifact declared in the manifest exists and its declared digest
   matches.
5. If signing is required, a valid PEM certificate and a detached signature are
   present.
6. SBOM / provenance files declared in `security` exist.

```bash
rosclaw hub verify ./path/to/asset
rosclaw hub verify ./path/to/asset --no-signature --json
```

### Checksum file format

`checksums.txt` is a newline-delimited list of file digests:

```text
sha256:<hexdigest>  <relative/path>
```

Two spaces separate the digest and the relative path. Example:

```text
sha256:6b4ffdb036f30ddd0adb5c3b1fed6483e64abe3c3711b9aa8e3405f2dcaea444  artifacts/skill/behavior_tree.xml
sha256:e43cf11f0181ccb99d92e54d51df5170a60b9ccc8a81ea2ce6335dc4517a0437  artifacts/models/policy.safetensors
```

### Signature placeholders

The current release uses a **placeholder** HMAC-SHA256 signature and a dummy
certificate so the verifier and installer can exercise the full signing
pipeline without requiring Sigstore network access. The signing key is
hardcoded for testing and must be replaced with production material before
connecting to a real registry.

## Permission policy (`src/rosclaw/hub/permissions.py`)

Permissions declared in `manifest.yaml` are evaluated by the installer unless
`--allow-real-robot`, `--allow-safety-config-changes`, or
`--allow-network-inbound` is passed.

| Permission category | Examples | Dangerous? |
|---------------------|----------|------------|
| `hardware.real_robot_execution` | Move real motors | Yes — requires `--allow-real-robot` |
| `hardware.actuators` | arms, gripper, legs | Listed for audit |
| `ros.topics_write` | `/cmd_vel`, `/arm_controller/command` | Yes for motion topics |
| `filesystem.write` | `$ROSCLAW_HOME/runtime/mcp.d` | Yes if not under runtime tree |
| `network.inbound` | `localhost` vs `0.0.0.0` | Yes if non-local |
| `modifies.safety_config` | Joint limits, collision geometry | Yes — requires `--allow-safety-config-changes` |
| `requires_human_approval` | Real robot, body.yaml, mcp_config | Blocks unattended install unless `--yes` |

Check policy locally:

```bash
rosclaw hub policy check ./path/to/asset --json
```

## License policy (`src/rosclaw/hub/licenses.py`)

License checks enforce:

- The `spdx` identifier is in the allowed whitelist (OSI-approved + custom
  ROSClaw licenses).
- A `license_file` exists in the asset directory.
- `commercial_use`, `redistribution`, `export_control`, and `data_rights`
  declarations match policy.
- If `requires_acceptance` is implied by a non-standard license, the user must
  pass `--accept-license` or `--yes`.

## Secret scanning (`src/rosclaw/hub/publisher.py`)

Before any publish, the publisher scans the asset directory for:

- Private keys (RSA, OpenSSH, EC, DSA, PGP)
- AWS access / secret keys
- API keys, bearer tokens, generic secrets
- Password literals

By default, any finding fails the publish. In warning mode the publish succeeds
but the finding is recorded.

## Install-time guards

The installer (`src/rosclaw/hub/installer.py`) performs an atomic transaction:

1. Acquire `assets.lock`.
2. Verify asset integrity.
3. Check compatibility (OS, arch, Python, ROS, robot).
4. Check permission and license policy.
5. Resolve dependencies.
6. Copy artifacts to `~/.rosclaw/hub/installed/`.
7. Write runtime registry JSON.
8. Optionally merge MCP config.
9. Run health checks.
10. Save the lockfile entry and installed-state record.

If any step fails, staged files are removed via `_rollback()`.

## Network policy

The manifest `permissions.network` section declares:

- `outbound`: allowed destinations (`robot-local`, `internet`, etc.)
- `inbound`: allowed bind addresses

Non-local inbound access is denied unless `--allow-network-inbound` is
provided.

## Audit trail

Every install, update, and uninstall records:

- The canonical `rosclaw://` reference
- Source path or registry URL
- Lifecycle and health status
- Dependency graph
- Timestamps in the lockfile and installed-state JSON

```bash
rosclaw hub list --installed --json
```

## Future hardening

- Replace placeholder signing with Sigstore / cosign verification.
- Add TUF metadata (`root.json`, `timestamp.json`, `snapshot.json`) to the fake
  registry.
- Sandbox every install verification step inside an isolated process.
- Add provenance attestation verification against builder identity.
- Add secret scanning for model weight files and binary artifacts.

## See also

- [CLI reference](cli.md)
- [Publish guide](publish_guide.md)
- [Asset manifest reference](asset_manifest.md)
- `src/rosclaw/hub/verifier.py`
- `src/rosclaw/hub/permissions.py`
- `src/rosclaw/hub/licenses.py`
- `tests/hub/test_security_regression.py`
