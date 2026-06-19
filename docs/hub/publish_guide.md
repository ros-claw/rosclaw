# Publishing ROSClaw Hub Assets

This guide covers how to prepare, verify, sign, bundle, and publish a ROSClaw
Hub asset. Assets include `skill`, `provider`, `hardware_mcp`, `digital_twin`,
and `cognitive_wiki`.

## Prerequisites

- A valid `manifest.yaml` (see [asset_manifest.md](asset_manifest.md)).
- Artifact files referenced in the manifest exist in the asset directory.
- A registry profile if you plan to upload (`rosclaw hub login`).

## Prepare an asset directory

A minimal skill asset looks like this:

```text
my_skill/
├── manifest.yaml
├── LICENSE
├── artifacts/
│   └── skill/
│       ├── behavior_tree.xml
│       └── prompt.md
└── signatures/   # created by publish --sign
    ├── cert.pem
    └── signature.bin
```

The `security` section in `manifest.yaml` tells the publisher what to generate:

```yaml
security:
  signing:
    required: true
    scheme: sigstore
    certificate: signatures/cert.pem
  checksums:
    algorithm: sha256
    file: checksums.txt
  sbom: SBOM.spdx.json
  provenance: PROVENANCE.json
```

## Dry-run publish

Validate and secret-scan without writing anything:

```bash
rosclaw hub publish ./my_skill --dry-run
```

Output on success:

```text
Dry-run; no bundle or registry was written.
```

If the secret scanner finds a key or password, the dry-run fails with the file
and line number.

## Create a signed bundle

Generate `checksums.txt`, SBOM, provenance, placeholder signing material, and a
`.rosclaw` tar.gz bundle:

```bash
rosclaw hub publish ./my_skill --private --sign --output ./dist
```

The output bundle path will be similar to:

```text
./dist/my_namespace-my_skill-1.2.0.rosclaw
```

Bundle contents:

```text
manifest.yaml
checksums.txt
SBOM.spdx.json
PROVENANCE.json
signatures/cert.pem
signatures/signature.bin
artifacts/skill/behavior_tree.xml
artifacts/skill/prompt.md
```

## Publish to a registry

```bash
rosclaw hub login --registry http://localhost:8787 --token $TOKEN --insecure-local
rosclaw hub publish ./my_skill --private --sign --registry http://localhost:8787
```

If the registry client is available, the publisher uploads the bundle and the
registry records:

- The manifest in `manifests/<type>/<namespace>/<name>/<version>.yaml`
- The bundle in `bundles/<type>/<namespace>/<name>/<version>.rosclaw`
- A catalog entry in `catalog.jsonl`

## Checksums and provenance

The publisher computes sha256 digests for `manifest.yaml` and every declared
artifact, then writes `checksums.txt`:

```text
sha256:<manifest-digest>  manifest.yaml
sha256:<artifact-digest>  artifacts/skill/behavior_tree.xml
```

The provenance file is generated twice:

1. First with a placeholder bundle digest so the file exists during bundling.
2. After the bundle is created, the real bundle digest is computed and the
   provenance file is rewritten.
3. The bundle is then recreated so it contains the final provenance.

This two-phase build guarantees that the provenance attestation matches the
actual shipped bundle.

## Private vs public visibility

- `--private` sets `visibility.scope = private`.
- `--public` sets `visibility.scope = public`.

Private assets are filtered by registry access control. The local installer does
not enforce visibility; the registry does.

## Secret scanning modes

Default: findings are fatal.

To record warnings without blocking publish, set `secret_scan_fail_on_find`
via the Python API or adjust the CLI in the future when a flag is added.

## Recommended CI pipeline

```yaml
- name: Validate manifest
  run: rosclaw hub validate ./assets/my_skill/manifest.yaml --json
- name: Dry-run publish
  run: rosclaw hub publish ./assets/my_skill --dry-run
- name: Publish bundle
  run: rosclaw hub publish ./assets/my_skill --private --sign --registry $REGISTRY
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Declared artifact missing on disk` | Add the missing file or remove it from `artifacts` |
| `Secret scan failed` | Remove keys, tokens, or passwords from the asset directory |
| `No registry client available` | Run `rosclaw hub login` or pass `--registry` |
| `Manifest validation failed` | Use `rosclaw hub validate` to get field-level errors |

## See also

- [CLI reference: `rosclaw hub publish`](cli.md#publish-command)
- [Security model](security.md)
- [Asset manifest reference](asset_manifest.md)
- `src/rosclaw/hub/publisher.py`
- `tests/hub/test_publisher.py`
- `tests/hub/test_security_regression.py`
