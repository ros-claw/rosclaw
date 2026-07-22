# Hub fixture key

`fixture-private.pem` is an intentionally public, test-only Ed25519 key. It is
scoped to fixture manifests through `trust.json` and is never included in the
packaged ROSClaw trust store. Never use it to sign a release asset.
