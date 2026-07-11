# LeRobot Dataset Export Guide

This document describes how to export ROSClaw Practice episodes into real
`LeRobotDataset` v3 format using the P2 / P2.1 Gate A dataset writer.

## Input Practice episode format

P2.1 accepts either:

1. An episode directory:

   ```text
   episode_dir/
     episode.json
     frames/
       front_000000.png
       front_000001.png
       ...
   ```

2. A single `episode.json` file whose image paths are relative to the JSON's
   parent directory.

The `episode.json` schema is the ROSClaw Practice normalized format
(`rosclaw.practice.normalized.v2`). See
`examples/practice/minimal_lerobot_episode/episode.json` for a minimal example,
`examples/practice/rich_lerobot_episode/episode.json` for a Gate A rich example
with safety/failure/intervention metadata, and
`examples/practice/physical_lerobot_episode/episode.json` for a Gate B example
with physical telemetry.

## NormalizedPracticeEpisode schema (v2)

```json
{
  "schema_version": "rosclaw.practice.normalized.v2",
  "episode_id": "minimal_lerobot_episode_000001",
  "robot": {
    "robot_id": "mock_robot",
    "body_profile": "mock_body",
    "body_yaml_path": "/path/to/body.yaml",
    "body_hash": "mock_body_hash_abc123"
  },
  "task": {
    "text": "transfer the cube",
    "task_id": "transfer_cube"
  },
  "fps": 10,
  "environment": "mock_tabletop",
  "operator": "test_operator",
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "observation": {
        "state": [0.0, 0.0, 0.0],
        "images": {"front": "frames/front_000000.png"}
      },
      "action": [0.0, 0.0, 0.0],
      "done": false,
      "success": false,
      "safety": {
        "decision": "ALLOW",
        "modified": false,
        "risk_score": 0.1,
        "reason_code": null
      },
      "failure": {
        "active": false,
        "code": "NONE",
        "severity": 0
      },
      "intervention": {
        "active": false,
        "source": "NONE",
        "confidence": null
      },
      "action_context": {
        "source": "POLICY",
        "was_clamped": false
      },
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

v1 episodes are automatically migrated to v2 with unknown/inactive defaults for
missing fields.

## LeRobotDataset export command

```bash
rosclaw lerobot export-dataset \
  --episode examples/practice/minimal_lerobot_episode \
  --output /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal \
  --fps 10 \
  --profile minimal
```

Or through the practice export path:

```bash
rosclaw practice export \
  --format lerobot \
  --writer real \
  --profile minimal \
  --episode examples/practice/minimal_lerobot_episode \
  --output /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal \
  --fps 10
```

### Profiles

| Profile | Feature groups | Use case |
|---------|----------------|----------|
| `minimal` (default) | none | P2 backward compatibility. |
| `safety` | `safety` | Sandbox decision / risk metadata. |
| `physical` | `safety`, `action`, `physical_telemetry` | Physical telemetry + action provenance + safety. |
| `safety-rich` | `safety`, `failure`, `intervention`, `action`, `outcome` | Full Gate A metadata. |

Add individual groups on top of a profile with `--include <groups>`:

```bash
rosclaw practice export --format lerobot --writer real --profile safety \
  --include failure,intervention ...
```

### Optional flags

- `--allow-partial`: allow export when the requested profile cannot be fully
  satisfied.
- `--missing-policy {error,drop-frame,fill-last,nan}`: how to handle missing
  physical telemetry values. Default `nan` fills missing float readings with
  `NaN` and missing contact flags with `0`.
- `--visual-storage-mode {auto,images,videos}`: control image storage. Default
  `auto` uses per-frame images for short episodes and videos for long ones.
- `--use-videos`: legacy alias for `--visual-storage-mode videos`.
- `--include-body-snapshot`: copy body YAML / EMBODIMENT.md / calibration into
  `meta/rosclaw/body_snapshots/`.
- `--body-snapshot-mode {none,sanitized,full}`: default `sanitized` strips
  serial numbers, IP addresses, tokens, and geo coordinates.
- `--dry-run`: preview inferred features and warnings without writing a dataset.
- `--dataloader`: run a DataLoader smoke test after writing.
- `--robot-id`, `--body-profile`: override robot metadata.
- `--timeout-sec`: worker timeout (default: 300s).

## Feature mapping table

| ROSClaw Practice | LeRobotDataset | P2.1 Status |
|------------------|----------------|-------------|
| `frame.observation.state` | `observation.state` | supported |
| `frame.action` | `action` | supported |
| `task.text` | `task` | supported |
| `frame.timestamp` | timestamp / index metadata | supported |
| `frame.observation.images.<camera>` | `observation.images.<camera>` | supported |
| `frame.safety.*` | `rosclaw.sandbox.*` | supported |
| `frame.failure.*` | `rosclaw.failure.*` | supported |
| `frame.intervention.*` | `rosclaw.intervention.*` | supported |
| `frame.action_context.*` | `rosclaw.action.*` | supported |
| `frame.done` / `frame.success` | `rosclaw.done` / `rosclaw.success` | supported |
| `frame.observation.motor_current` | `observation.motor_current` | supported (Gate B) |
| `frame.observation.joint_temperature` | `observation.joint_temperature` | supported (Gate B) |
| `frame.observation.force_torque` | `observation.force_torque` | supported (Gate B) |
| `frame.observation.contact` | `observation.contact` | supported (Gate B) |
| `frame.observation.joint_velocity` | `observation.joint_velocity` | supported (Gate B) |
| `frame.observation.joint_effort` | `observation.joint_effort` | supported (Gate B) |
| depth | `observation.depth.<camera>` | planned (Gate C) |
| memory/how annotations | metadata | planned (P3) |

Categorical string fields (`decision`, `source`, `code`) are encoded as `int8` /
`int16` and mapped back to labels through `meta/rosclaw/vocab.json`.

## ROSClaw sidecars

Every ROSClaw-rich export writes the following files under `meta/rosclaw/`:

| Sidecar | Purpose |
|---------|---------|
| `schema.json` | Extension schema and exported feature keys. |
| `vocab.json` | Integer-to-label vocabularies for categorical features. |
| `episodes.parquet` | Episode-level summary (success, failure code, intervention counts). |
| `events.parquet` | One row per event per frame for decisions, failures, interventions, outcomes. |
| `sync_stats.parquet` | Per-episode timing/sync quality statistics (Gate B). |
| `units.json` | SI units for telemetry features (Gate B). |
| `feature_names.json` | Human-readable names and axis semantics (Gate B). |
| `body_snapshots/manifest.json` + `body.yaml` | Body snapshot manifest and sanitized body data. |

## Timing metadata and basic synchronization diagnostics (Gate B)

Frames may include the following optional timing fields:

> This section covers the timing metadata recorded in Gate B.  Full multi-rate
> canonical timeline construction, per-feature resampling, provenance, and
> synchronization quality gates are implemented in Gate B.1.

- `source_timestamp_ns`: sensor-source timestamp in nanoseconds.
- `clock_domain`: clock domain label (e.g. `ros_time`, `realtime`).
- `episode_time_sec`: monotonic episode-relative time in seconds.

`sync_stats.parquet` reports the clock domain, start/end source timestamps,
frame deltas, and missing timestamp counts for each episode.

Load and index validation:

```bash
rosclaw lerobot validate-dataset \
  --dataset /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal
```

Rich validation (also checks ROSClaw sidecars and features):

```bash
rosclaw lerobot validate-dataset \
  --dataset /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal \
  --level rich
```

DataLoader smoke test:

```bash
rosclaw lerobot smoke-dataloader \
  --dataset /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal \
  --json
```

Expected output:

```text
[rosclaw-lerobot] Dataset validation
  Status:      ok
  Load OK:     True
  Index OK:    True
  Frames:      3
  Episodes:    1
```

## ROSClaw sidecars

A P2.1 export writes extra files under `meta/rosclaw/` that standard
LeRobotDataset ignores but ROSClaw tooling can consume:

| File | Content |
|------|---------|
| `meta/rosclaw/schema.json` | Extension schema declaring ROSClaw feature groups. |
| `meta/rosclaw/vocab.json` | Categorical label -> integer code mappings. |
| `meta/rosclaw/episodes.parquet` | Episode-level metadata (success, failure code, intervention count, etc.). |
| `meta/rosclaw/body_snapshots/` | Optional sanitized body configuration snapshot. |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `runtime_not_configured` | No LeRobot runtime configured. | Run `rosclaw setup lerobot --profile core --mode isolated`. |
| `dataset_create_failed` | `LeRobotDataset.create` API mismatch. | Run `rosclaw lerobot dataset-api` and check the signature. |
| `image_shape_mismatch` | Frames have different image sizes. | Resize or crop frames to a common size before export. |
| `state_dim_mismatch` | State vectors have varying length. | Ensure each frame has the same state dimension. |
| `dataset_load_failed` | `HF_HUB_OFFLINE` not set or missing `finalize()`. | Worker sets offline mode automatically; ensure the runtime is LeRobot >= 0.6.x. |
| `image_file_not_found` | Episode image paths are wrong or images are missing. | Check that paths are relative to the JSON file's parent directory. |

## P2.1 Gate A limitations

- Single RGB camera is the default stable path. Multi-camera is supported by the
  writer but not yet exposed through CLI profiles (Gate C).
- Physical telemetry and depth are reserved for Gate B / Gate C.
- No push-to-Hub in Gate A.

## Roadmap

- **Gate B**: physical telemetry (`observation.motor_current`,
  `observation.joint_temperature`, `observation.force_torque`,
  `observation.contact`) and sync quality metadata.
- **Gate C**: multi-camera / depth CLI, video hardening, and depth encoding.
- **P3**: memory/how annotations, reward columns, Hub upload.
