# ROSClaw Hub Security Model

The ROSClaw Hub distributes executable physical-AI assets. A malicious or
accidentally dangerous asset could expose credentials, alter configuration, or
make hardware capabilities available to an Agent. The Hub therefore combines
schema validation, complete payload integrity, independent publisher trust,
permission and license policy, bounded extraction, secret scanning, and
transaction rollback.

These controls secure asset preparation and installation. They do not replace
`rosclawd` runtime authorization, controller watchdogs, physical E-Stop, or
body-specific hardware acceptance.

## Threat model

| Threat | Current mitigation and boundary |
|--------|---------------------------------|
| Payload changed after signing | sha256 coverage of every regular payload file plus a detached Ed25519 signature over the exact manifest and checksum bytes |
| Registry substitutes another valid asset | catalog digest/identity validation plus byte-identical synchronized, fetched, and bundled manifests bound to the resolved canonical reference |
| Untrusted publisher key | independent trust store with active status and canonical-reference scopes |
| Archive path traversal or link escape | version-independent bounded extractor rejects absolute/parent paths, links, special files, duplicates, and file-parent conflicts |
| Leaked text credentials | publish-time pattern scan; this is a guard, not proof that binary payloads contain no secrets |
| Unauthorized real-robot capability | install defaults to deny and requires `--allow-real-robot`; runtime execution still requires daemon authorization |
| Safety configuration changes | explicit `--allow-safety-config-changes` gate |
| Non-local inbound listener | explicit `--allow-network-inbound` gate |
| Failed or changed installation copy | partial copies are removed; the copied manifest must match the verified snapshot and the complete target tree is verified again before registry mutation |
| Catalog rollback or freeze | Not yet solved; TUF-style root, snapshot, timestamp, and rollback protection remain required for a public registry |

## Verification (`rosclaw hub verify`)

The verifier checks an asset directory without executing asset code:

1. Reject a symlinked root, symlinks, special filesystem entries, and control-character paths beneath it.
2. Load and validate `manifest.yaml` with the Pydantic schema.
3. Require sha256 and parse every checksum line strictly.
4. Reject malformed, duplicate, absolute, parent-traversing, and backslash paths.
5. Verify every listed file and reject regular payload files not covered by the checksum list.
6. Verify every declared artifact digest and required SBOM/provenance file.
7. Load the signing key from an independent trust store, require an active
   Ed25519 key whose scope matches the canonical asset reference, and verify the
   detached signature.

```bash
rosclaw hub verify ./path/to/asset \
  --trust-store /etc/rosclaw/hub-trust.json
rosclaw hub verify ./path/to/asset \
  --trust-store /etc/rosclaw/hub-trust.json \
  --json
```

`--no-signature` is an explicit local-development escape hatch. It emits a
warning and must not be used for registry or production installation. A
manifest cannot disable signature verification requested by the caller.

### Checksum format

`checksums.txt` uses one sha256 digest per line, with exactly two spaces before
the relative path:

```text
sha256:6b4ffdb036f30ddd0adb5c3b1fed6483e64abe3c3711b9aa8e3405f2dcaea444  manifest.yaml
sha256:e43cf11f0181ccb99d92e54d51df5170a60b9ccc8a81ea2ce6335dc4517a0437  artifacts/models/policy.safetensors
```

The checksum file and detached signature are excluded to avoid self-reference.
Every other regular file must be listed.

### Detached Ed25519 signature

The signed bytes are:

```text
"ROSCLAW-HUB-ASSET-SIGNATURE-V1\0"
+ exact manifest.yaml bytes
+ "\0"
+ exact checksums.txt bytes
```

The signature file contains base64-encoded 64-byte Ed25519 signature material.
The public key is never trusted because it appears in an asset. Instead,
`security.signing.key_id` selects a 32-byte Ed25519 public key from an
operator-controlled trust store.

```json
{
  "schema_version": "rosclaw.hub.trust.v1",
  "keys": {
    "publisher-release-2026": {
      "algorithm": "ed25519",
      "public_key_base64": "<base64 public key>",
      "status": "trusted",
      "scopes": ["rosclaw://skill/publisher/*@*"]
    }
  }
}
```

Trust-store precedence is `--trust-store`, then `ROSCLAW_HUB_TRUST_STORE`, then
the packaged `rosclaw/hub/trust/keys.json`. The packaged store is intentionally
empty. Test keys exist only under `tests/fixtures/hub_keys/`.

## Safe bundle extraction

Remote `.rosclaw` archives are fully validated before the first member is
written. The extractor applies the same behavior on Python 3.11, 3.12, and
3.13 and enforces default limits of 10,000 members and 2 GiB uncompressed data.
It accepts only regular files and directories and removes set-user-ID and
set-group-ID permission bits. Invalid archives return
`HUB_INDEX_VERIFY_FAILED`, and staging directories are removed on success and
failure.

## Permission policy

`src/rosclaw/hub/permissions.py` evaluates declared installation capability.
High-risk categories require their own explicit flags; `--yes` does not grant
them.

| Permission category | Installation behavior |
|---------------------|-----------------------|
| `hardware.real_robot_execution` | denied unless `--allow-real-robot` is present |
| `modifies.safety_config` | denied unless `--allow-safety-config-changes` is present |
| non-local `network.inbound` | denied unless `--allow-network-inbound` is present |
| actuator and ROS motion declarations | retained as dangerous-permission audit labels |
| sensitive filesystem writes | retained as dangerous-permission audit labels |

Installing an asset only makes it available. Actual physical execution must
still pass the daemon's body binding, Permit, Session, Action Lease, safety,
and executor checks.

Inspect policy without installing:

```bash
rosclaw hub policy check ./path/to/asset --json
```

## License policy

License checks enforce the accepted SPDX set, declared license-file presence,
commercial and redistribution terms, export-control declarations, and data
rights. A license that requires acceptance needs `--accept-license` or `--yes`.
Neither flag grants real-robot, safety-config, or inbound-network permission.

## Secret scanning

The publisher scans UTF-8 text files for private-key blocks, AWS credentials,
API keys, bearer tokens, generic secrets, and password literals. Findings fail
a non-dry-run publish by default and are warnings in dry-run mode. Signing keys
must remain outside the asset root.

Pattern scanning can miss encoded or binary secrets. Production publishing
should add organization-specific scanners and artifact provenance controls.

## Remote install binding

For a registry reference, installation performs the following checks before
writing installed state:

1. Resolve an immutable concrete version from the synchronized local catalog.
2. Require catalog reference, manifest digest, and manifest identity to agree;
   rebuild searchable metadata from that manifest instead of trusting duplicate
   catalog fields.
3. Require the fetched manifest to match the synchronized bytes and identity.
4. Safely extract the fetched bundle into a private staging directory.
5. Require the bundled manifest to match both the fetched bytes and requested
   canonical reference.
6. Run complete payload, signature, compatibility, license, permission, and
   dependency checks through the local installer.
7. Copy files, require the copied manifest to match the verified snapshot, and
   verify the complete copied tree again before updating runtime registries
   transactionally. Partial copies are removed on failure.

Remote `--dry-run` performs steps 1 through 6 and skips step 7.

## State and audit boundary

The Hub lockfile and installed-state JSON record the active installation,
source, dependencies, health, and timestamps. Uninstall removes active state;
this is not an immutable security audit log. Physical actions are audited by
the separate `rosclawd` receipt and durable-ledger path.

## Remaining production hardening

- Add TUF-style signed root, targets, snapshot, and timestamp metadata with
  rollback and freeze protection.
- Establish public key ownership, offline root, rotation, revocation, and
  incident-response governance before populating the packaged trust store.
- Verify provenance attestations against an approved builder identity instead
  of checking only file integrity.
- Add transparency or an external append-only witness for public releases.
- Replace the local/file registry stub with a production client and anonymous
  verified download path.
- Expand secret and malware scanning to model weights and binary artifacts.

Until those items exist, the generic Hub is a verified local developer asset
pipeline, not a production public software-supply-chain service.

## See also

- [CLI reference](cli.md)
- [Publish guide](publish_guide.md)
- [Asset manifest reference](asset_manifest.md)
- `src/rosclaw/hub/verifier.py`
- `src/rosclaw/hub/_compat.py`
- `tests/hub/test_signature_trust.py`
- `tests/hub/test_archive_security.py`
