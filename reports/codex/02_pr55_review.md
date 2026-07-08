# PR #55 Review

PR #55 local branch: `codex/pr55-review`.

Base used for audit: `origin/main` at `0916efc0a8d6c90a104e14d43ca7f594d478e132`.

Initial PR head used for audit: `4a28d0f6bd13efcae661729ed440d9074ca7244e`.

Latest PR head before pushing this follow-up fix: `0ba634b1a891331560d1085a8020d1625d0a4d47`.

## Diff Summary

The initial PR diff reported 41 files changed, 6560 insertions, 36 deletions.

This follow-up commit adds the audit reports, validation script, RH56 fixture, fixture-record CLI path, stricter verifier coverage, and regression fixes on top of the latest PR source branch.

Primary areas:

- Practice schema/API: `src/rosclaw/practice/ids.py`, `schemas.py`, `config.py`
- Storage/catalog/artifacts: `artifact_store.py`, `storage/catalog.py`, `storage/layout.py`
- Recorder/coordinator/writers: `recorder.py`, `coordinator.py`, `writers/mcap_writer.py`
- Strict verification: `verifier.py`
- Distillation: `distiller.py`
- SeekDB ingest/query: `seekdb_ingestor.py`, `query.py`, `memory/seekdb_client.py`
- Export: `exporters/parquet_exporter.py`, `exporters/lerobot_exporter.py`
- CLI wiring: `src/rosclaw/cli.py`
- Practice tests: `tests/practice/*`

The full file list is recorded in `reports/codex/pr55_files.txt`.

## What PR #55 Gets Right After This Pass

- Practice v2 catalog and artifact store have real filesystem and SQLite-backed behavior.
- Artifact manifest verification detects sha256 tampering.
- Strict verify now catches missing raw event envelope fields.
- Distill writes episode summaries and derived artifacts without replacing raw event logs.
- Local SQLite-backed ingest/query path works for failures, body cognition, sim2real deltas, skill candidates, and interventions.
- Parquet and LeRobot exporters generate real files when `practice-export` dependencies are installed.
- The deterministic RH56 fixture exercises the core Practice lifecycle through the same `RuntimeBus -> PracticeRecorder` path as runtime events.

## Regressions Fixed During Review

- Optional external `rosclaw_rh56` import no longer breaks Practice collection on machines without the RH56 runtime package.
- SeekDB/query backend connection failures now return actionable CLI errors instead of tracebacks.
- Root `.mcp.json` and `mcp.json` are not required as tracked docs assets because `.gitignore` intentionally marks them as local state.
- Skill template tests no longer fail because compiled bytecode was generated inside template directories.
- Memory records with generated ids can be retrieved by the generated id.

## Hidden Unicode

Command run over git-tracked `.py`, `.md`, `.yaml`, `.yml`, `.json`, and `.toml` files:

```bash
python - <<'PY'
from pathlib import Path
bad = []
for p in Path(".").rglob("*"):
    if p.is_file() and p.suffix in {".py",".md",".yaml",".yml",".json",".toml"}:
        s = p.read_text(errors="ignore")
        for i, ch in enumerate(s):
            if ord(ch) in list(range(0x202A,0x202F)) + list(range(0x2066,0x2070)):
                bad.append((str(p), i, hex(ord(ch))))
if bad:
    raise SystemExit(1)
print("OK: no bidi control chars found in git tracked source/docs")
PY
```

Result: pass.

## SeekDB Audit

Evidence:

- Docker socket smoke for `localhost:2881` passed.
- `rosclaw practice ingest-seekdb` currently accepts `--seekdb-path` and creates/uses a SQLite database through `SeekDBSQLiteClient`.
- `rosclaw practice query ...` uses the same local SQLite backend.

Gap:

- There is no verified `--seekdb-url http://localhost:2881` ingest/query path.
- The task's "real SeekDB/OceanBase container" gate is therefore not met.

## Artifact Audit

Evidence:

- `ArtifactStore` writes raw JSONL, YAML summaries, Parquet artifacts, manifests, size, sha256, created timestamps, and mime/type metadata.
- Manual tamper test appended to a generated summary artifact and reran `practice verify --strict`; result was rc 1 with a sha256 mismatch.

## Review Verdict

PR #55 is materially improved and the local Practice closed loop is now real, repeatable, and tested. It is still not merge-ready against the task's explicit gates because repo format, Darwin CLI, ROS bridge Loop B, and real SeekDB 2881 integration remain incomplete.
