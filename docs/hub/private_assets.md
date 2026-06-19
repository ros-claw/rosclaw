# Private and Internal Assets

Not every ROSClaw Hub asset is public. Teams often need private skills,
internal hardware MCP servers, or pre-release digital twins. The Hub supports
private assets through visibility controls, local registries, and access-token
authentication.

## Visibility scope

The `visibility` section in `manifest.yaml` controls who can see an asset:

```yaml
visibility:
  scope: private          # public | private | internal
  allowed_orgs:
    - my-org
  allowed_users:
    - alice@example.com
```

| Scope | Behavior |
|-------|----------|
| `public` | Anyone can discover and install |
| `private` | Only authenticated users in `allowed_orgs` / `allowed_users` |
| `internal` | Same as private, but intended for org-internal registries |

The local installer does not enforce visibility; the registry filters search
and download results based on the caller's token.

## Publish a private asset

```bash
rosclaw hub publish ./my_internal_skill --private --sign --registry $REGISTRY
```

This sets `visibility.scope = private` and uploads the bundle. Only callers
with a valid token that maps to an allowed org or user can sync or install it.

## Run a local registry for testing

For development, use the fake local-file registry or start the test HTTP server:

```bash
python -m tests.fixtures.fake_registry.server --port 8787 &
rosclaw hub login --registry http://localhost:8787 --token fake-valid-token --insecure-local
```

`--insecure-local` allows plain HTTP and local file registries. Do not use it
for production.

## Internal asset naming

Use a namespace that identifies your organization:

```text
rosclaw://skill/my_org/internal_pick_place@1.0.0
```

Keep `namespace` stable; it is part of the canonical reference and the install
path.

## Offline install from a bundle

If a registry is not reachable, install directly from a local `.rosclaw` bundle
by extracting it and installing the directory:

```bash
tar -xzf my_skill-1.2.0.rosclaw -C /tmp/my_skill
rosclaw hub install /tmp/my_skill --yes
```

Or install from a local source directory before bundling:

```bash
rosclaw hub install ./my_skill --dry-run
rosclaw hub install ./my_skill --yes
```

## Token storage

Tokens are stored in `~/.rosclaw/config/hub_auth.json` with a JSON fallback for
testing. In production, migrate to the OS keyring or a secrets manager.

## Future: private registry hosting

- Authenticated S3 / GCS-backed registries
- OCI-compatible bundle storage
- Org-level namespace ACLs
- Audit logging for every sync, install, and publish

## See also

- [CLI reference: login / logout / publish](cli.md)
- [Publish guide](publish_guide.md)
- `src/rosclaw/hub/auth.py`
- `src/rosclaw/hub/client.py`
- `tests/hub/test_e2e_fake_registry.py`
