# Publishing ROSClaw Hub Assets

This guide covers how to validate, hash, sign, bundle, and publish a generic
ROSClaw Hub asset. Generic assets include `skill`, `provider`, `hardware_mcp`,
`digital_twin`, and `cognitive_wiki`.

The generic `rosclaw hub` registry is currently a local/file-backed developer
implementation. It is separate from the legacy MCP discovery endpoint used by
`rosclaw mcp`, and it is not yet a production public registry.

## Prerequisites

- A valid `manifest.yaml` (see [asset_manifest.md](asset_manifest.md)).
- Every artifact declared by the manifest exists beneath the asset root.
- An Ed25519 PKCS8 PEM private key stored outside the asset directory.
- A stable key ID and an independently distributed trust store.
- A registry profile if uploading to a local test registry.

Generate a development key with OpenSSL:

```bash
openssl genpkey -algorithm ED25519 -out /secure/path/hub-signing-key.pem
chmod 600 /secure/path/hub-signing-key.pem
```

Do not commit a production private key. The key under
`tests/fixtures/hub_keys/` is intentionally public and must only be used by
tests and local demonstrations.

## Prepare an asset directory

A minimal skill asset looks like this before publication:

```text
my_skill/
|-- manifest.yaml
|-- LICENSE
`-- artifacts/
    `-- skill/
        |-- behavior_tree.xml
        `-- prompt.md
```

Declare the signature and integrity files in `manifest.yaml`:

```yaml
security:
  signing:
    required: true
    scheme: ed25519
    key_id: my-org-hub-release-2026
    file: signatures/manifest.ed25519
  checksums:
    algorithm: sha256
    file: checksums.txt
  sbom: SBOM.spdx.json
  provenance: PROVENANCE.json
```

All paths must be relative, remain under the asset root, and be distinct from
each other and from artifact paths. The detached signature must be beneath
`signatures/` with an `.ed25519` suffix. Symbolic links, non-regular entries
such as FIFOs or sockets, backslashes, and control-character paths are
rejected. A real publish also rejects every missing declared artifact; dry-run
reports missing artifacts without producing a bundle.

## Configure the signing identity

Pass the private key and key ID explicitly:

```bash
rosclaw hub publish ./my_skill \
  --sign \
  --signing-key /secure/path/hub-signing-key.pem \
  --signing-key-id my-org-hub-release-2026 \
  --output ./dist
```

An explicit `--output` creates a local bundle without requiring registry
login. Add `--registry` (or log in to an active profile) only when uploading.

CI can use equivalent environment variables:

```bash
export ROSCLAW_HUB_SIGNING_KEY=/secure/path/hub-signing-key.pem
export ROSCLAW_HUB_SIGNING_KEY_ID=my-org-hub-release-2026
```

If `security.signing.required` is true, publication fails closed when the key,
key ID, or Ed25519 scheme is missing. Registry upload always requires a signed
bundle.

## Trust store

Consumers verify the detached signature against a trust store that is not
part of the asset:

```json
{
  "schema_version": "rosclaw.hub.trust.v1",
  "keys": {
    "my-org-hub-release-2026": {
      "algorithm": "ed25519",
      "public_key_base64": "<base64-encoded 32-byte Ed25519 public key>",
      "status": "trusted",
      "scopes": ["rosclaw://skill/my-org/*@*"]
    }
  }
}
```

Scopes use shell-style matching against the complete canonical asset
reference. Keep scopes as narrow as the publisher's responsibility permits.
Select a trust store using either mechanism:

```bash
export ROSCLAW_HUB_TRUST_STORE=/etc/rosclaw/hub-trust.json
rosclaw hub verify ./my_skill

rosclaw hub verify ./my_skill --trust-store /etc/rosclaw/hub-trust.json
```

The packaged trust store is intentionally empty until public key ownership,
rotation, revocation, and release governance are established.

## Dry-run publish

Run the same manifest, layout, secret-scan, digest, and signing preparation
without creating an output bundle or uploading to a registry:

```bash
rosclaw hub publish ./my_skill --dry-run --sign \
  --signing-key /secure/path/hub-signing-key.pem \
  --signing-key-id my-org-hub-release-2026
```

Secret findings are reported as warnings during dry-run. A non-dry-run publish
rejects them by default.

## Bundle contents

A signed `.rosclaw` bundle contains entries similar to:

```text
manifest.yaml
LICENSE
checksums.txt
SBOM.spdx.json
PROVENANCE.json
signatures/manifest.ed25519
artifacts/skill/behavior_tree.xml
artifacts/skill/prompt.md
```

`checksums.txt` covers every regular payload file except itself and the
detached signature. This includes the manifest, license, SBOM, provenance, and
all artifacts. Undeclared payload files and malformed or duplicate checksum
entries cause verification to fail.

The Ed25519 signature covers the exact manifest bytes and exact checksum-file
bytes under the `ROSCLAW-HUB-ASSET-SIGNATURE-V1` domain. The provenance subject
is the immutable manifest digest. A bundle cannot contain its own final digest
without creating a self-reference, so the outer bundle digest is reported by
the publisher and registry instead of being embedded in the bundle.

## Publish to a local registry

```bash
rosclaw hub login \
  --registry file:///tmp/rosclaw-registry \
  --token fake-valid-token \
  --insecure-local

rosclaw hub publish ./my_skill \
  --private \
  --sign \
  --signing-key /secure/path/hub-signing-key.pem \
  --signing-key-id my-org-hub-release-2026 \
  --registry file:///tmp/rosclaw-registry
```

The file registry records:

- `manifests/<type>/<namespace>/<name>/<version>.yaml`
- `bundles/<type>/<namespace>/<name>/<version>.rosclaw`
- one catalog entry in `catalog.jsonl`

Remote installation binds the synchronized manifest, fetched manifest,
bundled manifest, resolved canonical reference, complete payload hashes, and
trusted signature. `--dry-run` performs the same download and verification but
does not install files.

## Recommended CI sequence

```yaml
- name: Provision signing key
  shell: bash
  env:
    HUB_SIGNING_KEY_PEM: ${{ secrets.HUB_SIGNING_KEY_PEM }}
  run: |
    install -m 600 /dev/null "$RUNNER_TEMP/hub-signing-key.pem"
    printf '%s' "$HUB_SIGNING_KEY_PEM" > "$RUNNER_TEMP/hub-signing-key.pem"
- name: Validate manifest schema
  run: rosclaw hub validate ./assets/my_skill/manifest.yaml --json
- name: Prepare signed asset without upload
  run: rosclaw hub publish ./assets/my_skill --dry-run --sign --json
  env:
    ROSCLAW_HUB_SIGNING_KEY: ${{ runner.temp }}/hub-signing-key.pem
    ROSCLAW_HUB_SIGNING_KEY_ID: my-org-hub-release-2026
- name: Publish signed bundle
  run: rosclaw hub publish ./assets/my_skill --private --sign --registry "$REGISTRY"
  env:
    ROSCLAW_HUB_SIGNING_KEY: ${{ runner.temp }}/hub-signing-key.pem
    ROSCLAW_HUB_SIGNING_KEY_ID: my-org-hub-release-2026
```

`hub validate` is schema validation only. CI should also verify the prepared or
published bundle using an independently provisioned trust store.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Signed Hub publication requires an Ed25519 private key` | Pass `--signing-key` or set `ROSCLAW_HUB_SIGNING_KEY` |
| `Signing key ID does not match` | Align the CLI/environment key ID with `security.signing.key_id` |
| `Signature key is not trusted` | Add the correct public key to an independent scoped trust store |
| `Security control path collision` | Give manifest, checksums, signature, SBOM, provenance, and artifacts distinct paths |
| `Declared artifact missing on disk` | Add the missing file or remove it from `artifacts` |
| `Secret scan failed` | Remove keys, tokens, or passwords from the asset directory |
| `No registry client available` | Run `rosclaw hub login` or pass `--registry` |

## See also

- [CLI reference: `rosclaw hub publish`](cli.md#publish-command)
- [Security model](security.md)
- [Asset manifest reference](asset_manifest.md)
- `src/rosclaw/hub/publisher.py`
- `tests/hub/test_signature_trust.py`
- `tests/hub/test_security_regression.py`
