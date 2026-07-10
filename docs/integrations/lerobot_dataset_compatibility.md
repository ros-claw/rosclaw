# ROSClaw × LeRobot Dataset Compatibility Matrix

This matrix tracks which LeRobotDataset features are supported by the ROSClaw
bridge, which are planned, and the relevant profile or CLI flag to enable them.

| Feature | Status | Since | CLI / Profile | Notes |
|---------|--------|-------|---------------|-------|
| `observation.state` / `action` / `task` | supported | P2 | `minimal` (default) | Core LeRobotDataset features. |
| Single RGB camera | supported | P2 | `minimal` | `observation.images.<camera>`; default camera is `front`. |
| Safety / sandbox metadata | supported | P2.1 Gate A | `safety`, `physical`, `safety-rich` | `rosclaw.sandbox.*` int8/float32 features. |
| Failure metadata | supported | P2.1 Gate A | `safety-rich` | `rosclaw.failure.*` features. |
| Intervention metadata | supported | P2.1 Gate A | `safety-rich` | `rosclaw.intervention.*` features. |
| Action provenance | supported | P2.1 Gate A | `physical`, `safety-rich` | `rosclaw.action.*` features. |
| Episode outcome (`done`, `success`) | supported | P2.1 Gate A | `safety-rich` | `rosclaw.done` / `rosclaw.success`. |
| Episode sidecar (`episodes.parquet`) | supported | P2.1 Gate A | any | Written under `meta/rosclaw/`. |
| Vocabulary sidecar (`vocab.json`) | supported | P2.1 Gate A | any with ROSClaw groups | Maps categorical string labels to integer codes. |
| Extension schema (`schema.json`) | supported | P2.1 Gate A | any | Declares required/optional ROSClaw feature groups. |
| Body snapshot | supported | P2.1 Gate A | `--include-body-snapshot` | Copies `body.yaml` / `EMBODIMENT.md` / `calibration.yaml` under `meta/rosclaw/body_snapshots/`. |
| DataLoader smoke validation | supported | P2.1 Gate A | `--dataloader`, `rosclaw lerobot smoke-dataloader` | Iterates one batch in the LeRobot runtime. |
| Dry-run feature preview | supported | P2.1 Gate A | `--dry-run` | No dataset written; prints inferred features and warnings. |
| Multi-camera / depth | planned | P2.1 Gate C | — | Multiple RGB/depth cameras; see roadmap. |
| Physical telemetry (current/force/temp) | planned | P2.1 Gate B | — | Motor current, joint temperature, force/torque, contact. |
| Memory / how annotations | planned | P3 | — | Language annotations and memory references. |
| Push-to-Hub upload | planned | P3 | — | Upload dataset to HuggingFace Hub. |

## Profile quick reference

```bash
# No ROSClaw-rich features (P2 behavior, default)
rosclaw practice export --format lerobot --writer real --profile minimal ...

# Sandbox / safety metadata only
rosclaw practice export --format lerobot --writer real --profile safety ...

# Safety + action provenance
rosclaw practice export --format lerobot --writer real --profile physical ...

# Full Gate A metadata
rosclaw practice export --format lerobot --writer real --profile safety-rich ...
```

Add optional feature groups on top of a profile:

```bash
rosclaw practice export --format lerobot --writer real --profile safety \
  --include failure,intervention ...
```

## Worker protocol versions

| Schema | Introduced | Compatibility |
|--------|------------|---------------|
| `rosclaw.lerobot.dataset_worker.v1` | P2 | Supported for legacy requests; missing fields default to P2 behavior. |
| `rosclaw.lerobot.dataset_worker.v2` | P2.1 Gate A | Adds `profile`, `feature_groups`, `vocab`, `visual_storage_mode`, dataloader validation, sidecar metadata. |

## Export report versions

| Schema | Introduced | Notes |
|--------|------------|-------|
| `rosclaw.lerobot.dataset_export.v1` | P2 | Basic target/dataset/validation blocks. |
| `rosclaw.lerobot.dataset_export.v1.1` | P2.1 Gate A | Adds `visual`, `runtime`, `lerobot_dataset_api`, `quality_gates`. |

`rosclaw lerobot doctor` reads the latest report and shows `validated` / `stale` /
`failed` / `not_configured` states.
