# N01 frozen SIM evidence

Physical source: `2dcb77bd03de034963776c7e2e89db5070d3c57e`.
Run ID: `f1e279292c684a769b569d84d9381c6a`.

`archive-manifest.json` binds each original relative path and SHA-256 to its
archived bytes. Files above 100 KB are gzip-compressed with fixed mtime.
`source-manifest.json` retains original local paths as provenance, not portable
lookup requirements. `acceptance-summary.json` is a compact derivative; the
full saved acceptance record and post-cleanup observations are in
`acceptance.json`. Practice episode and manifest are also included.

To reconstruct public replay input in a new directory, verify both hashes and
restore each file under `original_relative_path`. Do not overwrite the frozen
source directory. For example, from the repository root:

```bash
.venv/bin/python - <<'PYCODE'
import gzip, hashlib, json
from pathlib import Path
src = Path("docs/reports/ros-expert-harness/next/runs/n01-waffle-004")
out = Path("/tmp/n01-public-replay-input")
out.mkdir(exist_ok=False)
for row in json.loads((src / "archive-manifest.json").read_text())["inputs"]:
    data = (src / row["archive_path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["archive_sha256"]
    if row["gzip"]:
        data = gzip.decompress(data)
    assert hashlib.sha256(data).hexdigest() == row["original_sha256"]
    target = out / row["original_relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
for saved in (src / "inputs/actions").glob("*.verification.json"):
    (out / "actions" / saved.name).write_bytes(saved.read_bytes())
PYCODE
.venv/bin/python integrations/ros_probe/acceptance/coverage_audit.py \
  --directory /tmp/n01-public-replay-input --output /tmp/n01-public-replay-output
```

This is diagnostic replay of a frozen physical episode, not a new robot run.
Plan predictions cannot grant measured credit. Missing causal observations are
UNKNOWN. No authentication, action tickets or private ledger records are needed.
