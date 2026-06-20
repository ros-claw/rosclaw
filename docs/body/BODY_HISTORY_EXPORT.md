# ROSClaw Body History and Export

Every meaningful change to a body produces a **snapshot**: a point-in-time copy
of the effective body model plus a short fingerprint. Snapshots let you audit
when a body changed, compare states, and restore or transfer a configuration.

## What creates a snapshot

Snapshots are written automatically by:

- `rosclaw body init` / `rosclaw body create` / `rosclaw body link-eurdf`
- `rosclaw body update-state`
- `rosclaw body note` (when it triggers a recompile)

Each snapshot is stored under the selected body's `snapshots/` directory:

```text
bodies/<body_id>/
└── snapshots/
    ├── body-2026-06-20T12-34-56.123456.yaml
    ├── body-2026-06-20T12-34-56.123456.fingerprint
    ├── body-2026-06-20T13-12-34.789012.yaml
    └── body-2026-06-20T13-12-34.789012.fingerprint
```

Timestamps use microsecond precision (`%Y-%m-%dT%H-%M-%S.%f`) to avoid
collision during rapid changes. The `.fingerprint` file contains the effective
body hash at the time of the snapshot.

## Listing snapshots

```bash
rosclaw body history --workspace ~/.rosclaw
rosclaw body history --workspace ~/.rosclaw --json
```

Text output:

```text
============================================================
ROSClaw Body History
============================================================
Current hash: 4489f860c41b163a4498718ec2185058f4c72d5b50af3ce5ae88f900e73202e8
Snapshots: 3

  body-2026-06-20T12-34-56.123456.yaml  hash=4489f860c41b163...  size=4821 bytes
  body-2026-06-20T13-12-34.789012.yaml  hash=81f537ae5afc43ce...  size=4825 bytes
  body-2026-06-20T13-45-01.456789.yaml  hash=a1b2c3d4e5f67890...  size=4830 bytes
============================================================
```

JSON output contains an array of snapshot records:

```json
[
  {
    "timestamp": "2026-06-20T12-34-56.123456",
    "hash": "4489f860c41b163a4498718ec2185058f4c72d5b50af3ce5ae88f900e73202e8",
    "snapshot": "body-2026-06-20T12-34-56.123456.yaml",
    "size": 4821
  }
]
```

## Diffing against a snapshot

Use a snapshot as the baseline for `rosclaw body diff`:

```bash
rosclaw body diff --against snapshot:body-2026-06-20T12-34-56.123456.yaml
```

This compares the current effective body against the snapshot and reports
changes by category (`sensor_status`, `actuator_status`, `capability`,
`safety`, `structural`, `incident`, etc.), including whether the change
requires a skill compatibility recheck.

## Exporting a body

The `export` subcommand packages a body directory into a portable archive:

```bash
# Zip archive
rosclaw body export /backups/g1-real.zip --workspace ~/.rosclaw

# Tar archive
rosclaw body export /backups/g1-real.tar --format tar --workspace ~/.rosclaw

# Target directory (archive name is derived from body ID)
rosclaw body export /backups/ --format zip --workspace ~/.rosclaw
```

Exported archives have the body ID as the top-level directory:

```text
g1-real.zip
└── g1-real/
    ├── body.yaml
    ├── calibration.yaml
    ├── maintenance.log
    ├── refs/
    ├── snapshots/
    └── generated/
```

You can transfer the archive to another workspace and unzip it under
`bodies/<body_id>/`, or into another machine's `~/.rosclaw/bodies/`.

## Restoring from a snapshot

There is no single "restore" CLI command yet; restoration is a deliberate
manual operation because it may change skill compatibility and safety limits.

Recommended workflow:

1. Identify the snapshot you want to restore:

   ```bash
   rosclaw body history --json
   ```

2. Copy the snapshot over the current effective body:

   ```bash
   cp ~/.rosclaw/bodies/g1-real/snapshots/body-2026-06-20T12-34-56.123456.yaml \
      ~/.rosclaw/bodies/g1-real/refs/effective_body.json
   ```

3. Recompile and refresh artifacts:

   ```bash
   rosclaw body render --body g1-real
   rosclaw body validate --body g1-real
   rosclaw body inspect --body g1-real --skills
   ```

4. If the restored state is correct, create a new snapshot to record the
   restoration:

   ```bash
   rosclaw body note --body g1-real "Restored from snapshot body-2026-06-20T12-34-56.123456" \
     --type incident --severity warning
   ```

## Programmatic access

```python
from rosclaw.body.resolver import BodyResolver

resolver = BodyResolver(body_id="g1-real")

# Iterate snapshot files
for snap_path in sorted(resolver.snapshots_dir.glob("body-*.yaml")):
    fingerprint_path = snap_path.with_suffix(".fingerprint")
    hash_value = fingerprint_path.read_text().strip() if fingerprint_path.exists() else ""
    print(snap_path.name, hash_value)

# Create a snapshot manually
from rosclaw.body.schema import EffectiveBody
effective = resolver.get_effective_body()
resolver.create_snapshot(effective)
```

## Files

- `src/rosclaw/body/resolver.py` — `snapshots_dir`, `create_snapshot()`.
- `src/rosclaw/body/cli.py` — `history` and `export` subcommands.
- `src/rosclaw/body/diff.py` — `BodyDiffer.diff_against_snapshot()`.

## See also

- [BODY_REGISTRY.md](BODY_REGISTRY.md) — multi-body registry layout and CLI.
- [SKILL_COMPATIBILITY.md](SKILL_COMPATIBILITY.md) — why changes may trigger
  skill compatibility rechecks.
- [URI_SCHEME.md](URI_SCHEME.md) — `rosclaw://body/current/effective` and other
  stable references.
