# Private and Internal Hub Assets

The generic Hub can package private skills, providers, and hardware
integrations for local development. Its current registry client and fixture
server are not a production access-control service.

## Visibility metadata

The manifest can describe intended visibility:

```yaml
visibility:
  scope: private
  allowed_orgs:
    - my-org
  allowed_users:
    - alice@example.com
```

`public`, `private`, and `internal` are validated and indexed. The current
local/file registry does not enforce organization or user ACLs. Do not place a
sensitive asset in a registry directory readable by untrusted users.

## Sign a private asset

Private visibility does not weaken authenticity requirements. Keep the private
key outside the asset and scope the corresponding trust key to the private
namespace:

```bash
rosclaw hub publish ./my_internal_skill \
  --private --sign \
  --signing-key /secure/path/internal-hub-key.pem \
  --signing-key-id my-org-internal-2026 \
  --output ./dist
```

The consuming operator distributes a trust store separately:

```json
{
  "schema_version": "rosclaw.hub.trust.v1",
  "keys": {
    "my-org-internal-2026": {
      "algorithm": "ed25519",
      "public_key_base64": "<base64 public key>",
      "status": "trusted",
      "scopes": ["rosclaw://*/my-org/*@*"]
    }
  }
}
```

## Offline bundle install

Pass a `.rosclaw` bundle directly to the CLI. Do not pre-extract an untrusted
bundle with the system `tar` command:

```bash
rosclaw hub install ./dist/my-org-my-skill-1.2.0.rosclaw \
  --trust-store /etc/rosclaw/internal-hub-trust.json
```

ROSClaw validates the complete archive before writing the first member, then
performs manifest identity, payload, signature, license, permission, and
dependency checks. Staging is removed after success or failure.

A source directory can be checked in the same way:

```bash
rosclaw hub install ./my_internal_skill --dry-run \
  --trust-store /etc/rosclaw/internal-hub-trust.json
```

Add explicit `--allow-real-robot`, `--allow-safety-config-changes`, or
`--allow-network-inbound` only when the manifest requires that installation
capability and the operator has reviewed it.

## Local registry testing

Use a directory or the fixture HTTP server only in a controlled development
environment:

```bash
REGISTRY_DIR=$(mktemp -d)
touch "$REGISTRY_DIR/catalog.jsonl"
rosclaw hub login \
  --registry "file://$REGISTRY_DIR" \
  --token fake-valid-token \
  --insecure-local
```

`--insecure-local` permits a local path or plain HTTP fixture endpoint. It does
not add ACL enforcement.

## Credential boundary

The current `AuthStore` writes registry tokens to
`~/.rosclaw/config/hub_auth.json` using an atomic owner-only file (`0600`) in an
owner-only directory (`0700`) on POSIX. It rejects links, non-regular files, and
files owned by another user. Tokens remain plaintext and the store does not yet
use an OS keyring; production operators should prefer a keyring or external
secret manager once a production registry transport is available.

## Production gaps

- authenticated object storage or OCI transport;
- publisher and reader namespace ACL enforcement;
- OS-keyring or external secret-manager credentials;
- TUF-style catalog freshness and rollback protection;
- durable publish/sync/install audit events;
- governed public and organization trust-root rotation.

## See also

- [CLI reference](cli.md)
- [Publish guide](publish_guide.md)
- [Security model](security.md)
- `src/rosclaw/hub/auth.py`
- `src/rosclaw/hub/client.py`
