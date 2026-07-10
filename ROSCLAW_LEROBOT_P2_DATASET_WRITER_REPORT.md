# ROSClaw × LeRobot Bridge — P2 Dataset Writer Report

**Date:** 2026-07-10
**Scope:** Convert ROSClaw Practice episodes into real, loadable LeRobotDataset v3 via a runtime-isolated LeRobot worker.
**Runtime:** ROSClaw core on Python 3.11.15, LeRobot worker on Python 3.12.13 / LeRobot 0.6.1.

## 1. New and modified files

### New files

| File | Purpose |
|------|---------|
| `src/rosclaw/integrations/lerobot/practice_normalizer.py` | Validate and normalize Practice episodes to `NormalizedPracticeEpisode`. |
| `src/rosclaw/integrations/lerobot/dataset_worker_schema.py` | JSON request/response dataclasses for the dataset worker. |
| `src/rosclaw/integrations/lerobot/dataset_worker_main.py` | LeRobot-runtime-side worker entry point (no `rosclaw` imports). |
| `src/rosclaw/integrations/lerobot/dataset_worker_runner.py` | ROSClaw-side subprocess runner for dataset ops. |
| `src/rosclaw/integrations/lerobot/dataset_feature_infer.py` | Infer LeRobot feature schema from normalized episodes. |
| `src/rosclaw/integrations/lerobot/dataset_validator.py` | Delegate LeRobotDataset load/index validation to the worker. |
| `src/rosclaw/integrations/lerobot/dataset_report.py` | Export report persistence and validation-state helpers. |
| `tests/integrations/test_lerobot_practice_normalizer.py` | Normalizer contract tests. |
| `tests/integrations/test_lerobot_dataset_worker_schema.py` | Schema round-trip tests. |
| `tests/integrations/test_lerobot_dataset_feature_infer.py` | Feature inference tests. |
| `tests/integrations/test_lerobot_dataset_fake_worker.py` | Subprocess runner tests with a fake worker. |
| `tests/integrations/test_lerobot_dataset_export_cli.py` | CLI parsing and dispatch tests. |
| `tests/integrations/test_lerobot_dataset_validation_report.py` | Validation/report helper tests. |
| `tests/integrations/test_lerobot_dataset_no_runtime.py` | Graceful failure when no runtime is configured. |
| `examples/lerobot/sample_dataset_export_request.json` | Sample worker request. |
| `examples/lerobot/sample_dataset_export_report.json` | Sample export report. |
| `docs/integrations/lerobot_dataset_export.md` | New dataset export guide. |

### Modified files

| File | Change |
|------|--------|
| `src/rosclaw/integrations/lerobot/cli.py` | Added `export-dataset`, `validate-dataset`, `dataset-api` handlers and wired dataset export error reporting. |
| `src/rosclaw/cli.py` | Added argparse subparsers for `rosclaw lerobot export-dataset/validate-dataset/dataset-api`; fixed `practice export --writer real` fallback fps. |
| `src/rosclaw/integrations/lerobot/capabilities.py` | Updated `dataset_export_lerobot` description to reflect real dataset writer. |
| `src/rosclaw/integrations/lerobot/doctor.py` | Computes and reports latest dataset export validation state. |
| `src/rosclaw/integrations/lerobot/schemas.py` | Added `dataset_export_status` to `LeRobotDoctorReport`. |
| `docs/integrations/lerobot_bridge.md` | Added P2 section, updated capability list and architecture diagram. |
| `tests/integrations/conftest.py` | Added `fake_dataset_worker_script` fixture. |

## 2. CLI usage

### Export a real LeRobotDataset

```bash
rosclaw lerobot export-dataset \
  --episode examples/practice/minimal_lerobot_episode \
  --output /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal \
  --fps 10
```

Equivalent `practice export` path:

```bash
rosclaw practice export \
  --format lerobot \
  --writer real \
  --episode examples/practice/minimal_lerobot_episode \
  --output /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal \
  --fps 10
```

### Validate a dataset

```bash
rosclaw lerobot validate-dataset \
  --dataset /tmp/rosclaw_lerobot_dataset \
  --repo-id local/rosclaw_minimal
```

### Introspect the LeRobotDataset API

```bash
rosclaw lerobot dataset-api
```

## 3. NormalizedPracticeEpisode schema

```json
{
  "schema_version": "rosclaw.practice.normalized.v1",
  "episode_id": "minimal_lerobot_episode_000001",
  "robot": {
    "robot_id": "mock_robot",
    "body_profile": "mock_body",
    "body_yaml_path": null
  },
  "task": {
    "text": "transfer the cube",
    "task_id": "transfer_cube"
  },
  "fps": 10,
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
      "metadata": {}
    }
  ],
  "metadata": {
    "source_dir": "/absolute/path/to/episode_dir",
    "source": "rosclaw_practice"
  }
}
```

## 4. LeRobotDataset API introspection result

```text
LeRobotDataset API
  create signature: (repo_id: str, fps: int, features: dict, root: str | pathlib.Path | None = None,
                    robot_type: str | None = None, use_videos: bool = True, ...)
  add_frame:        yes
  save_episode:     yes
  consolidate:      no
  finalize:         yes
  LeRobot version:  0.6.1
```

The worker uses:

```python
LeRobotDataset.create(..., use_videos=False)
for frame in frames:
    dataset.add_frame(frame)   # frame includes "task"
dataset.save_episode()
dataset.finalize()
```

`consolidate()` is not available in this LeRobot build; `finalize()` is used instead.

## 5. Exported dataset directory structure

```text
/tmp/rosclaw_lerobot_dataset/
  meta/
    info.json
    stats.json
    tasks.parquet
  data/
    chunk-000/
      file-000.parquet
  rosclaw_export_report.json
```

## 6. LeRobotDataset load validation result

```text
[rosclaw-lerobot] Dataset validation
  Status:      ok
  Load OK:     True
  Index OK:    True
  Frames:      3
  Episodes:    1
  Sample keys: action, episode_index, frame_index, index,
               observation.images.front, observation.state, task, task_index, timestamp
  Image keys:  observation.images.front
```

## 7. Export report paths

- Per-export report: `<output_dir>/rosclaw_export_report.json`
- Persistent reports: `~/.rosclaw/lerobot/dataset_exports/<timestamp>_<repo_id>.json`
- Latest symlink: `~/.rosclaw/lerobot/dataset_exports/latest.json`

## 8. Test results

```text
env -u PYTHONPATH .venv/bin/python -m pytest tests/integrations -q --tb=short
102 passed, 1 skipped in ~9s
```

New tests cover:

- Practice episode normalization (valid, overrides, missing files, dim/shape mismatches).
- Dataset worker schema round-trips.
- Feature inference from normalized episodes.
- Fake worker export/inspect/validate paths.
- CLI dispatch for `export-dataset`, `validate-dataset`, `practice export --writer real/skeleton`.
- Report write/read and validation-state helpers.
- Graceful `runtime_not_configured` errors.

## 9. Current limitations

- Single RGB camera (`front`) only in P2.
- Depth, force/current, sandbox decisions, memory/how annotations are not exported.
- Multi-camera synchronization is not implemented.
- Push-to-Hug-Face-Hub is not implemented.
- Default stable visual path is `use_videos=False` + `dtype="image"`; video encoding is available but less tested.

## 10. P2.1 / P3 suggestions

- **P2.1:** multi-camera support, depth video, motor current / force-contact extra observations, `sandbox_decision` and `failure_label` metadata, human intervention markers.
- **P3:** reward/success columns, memory/how language annotations, push-to-Hub, benchmark integration, rollout data loop.

## Acceptance check

| Criterion | Status |
|-----------|--------|
| ROSClaw core does not import lerobot/torch | ✅ Verified by import-time checks |
| Practice episode normalizes to `NormalizedPracticeEpisode` | ✅ `normalize_practice_episode` tests pass |
| Real writer produces a loadable LeRobotDataset | ✅ `len(dataset) == 3` and `dataset[0]` accessible |
| Output dir contains real data/meta files | ✅ `meta/`, `data/` parquet produced |
| Export report written to output dir and `~/.rosclaw` | ✅ |
| Doctor shows last dataset export validation | ✅ `validated` state shown |
| Skeleton writer still works | ✅ CLI test passes |
| All integration tests pass | ✅ 102 passed, 1 skipped |
