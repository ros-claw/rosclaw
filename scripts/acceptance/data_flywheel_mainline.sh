#!/bin/bash
# DF-16A: Mainline Data Flywheel Acceptance — run the four gates locally and
# write reports/data_flywheel/mainline_acceptance.json.  Stop conditions
# (per the Phase II doc): knowledge collection errors, pyseekdb import
# errors, Memory v2 import errors, DataPlane init errors => FAIL.
# Gate C compares failures against reports/data_flywheel/env_baseline.json
# (host-environmental flakes); any NEW failure name fails the gate.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
OUT=reports/data_flywheel
mkdir -p "$OUT"
REPORT="$OUT/mainline_acceptance.json"
BASELINE="$OUT/env_baseline.json"
FAILED=""
export FAILED

echo "== Gate A: import / contract =="
$PY - > "$OUT/gate_a.json" 2>&1 <<'PYEOF'
import importlib, json
mods = {}
for name in ["rosclaw", "rosclaw.storage", "rosclaw.memory.v2", "rosclaw.knowledge",
             "rosclaw.how", "rosclaw.evolution", "rosclaw.darwin",
             "rosclaw_know", "rosclaw_know.contracts", "rosclaw_how", "pyseekdb"]:
    try:
        importlib.import_module(name)
        mods[name] = "ok"
    except Exception as exc:
        mods[name] = f"FAIL: {exc}"
vers = {}
for pkg, mod in [("pyseekdb", "pyseekdb"), ("rosclaw_know", "rosclaw_know"), ("rosclaw_how", "rosclaw_how")]:
    try:
        vers[pkg] = getattr(importlib.import_module(mod), "__version__", "?")
    except Exception:
        vers[pkg] = "unavailable"
print(json.dumps({"modules": mods, "versions": vers}))
PYEOF
grep -q "FAIL" "$OUT/gate_a.json" && FAILED="$FAILED gate_a"

echo "== Gate B: focused data-plane suites =="
$PY -m pytest tests/storage tests/memory tests/practice tests/how tests/knowledge tests/darwin tests/evolution -q -p no:cacheprovider > "$OUT/gate_b.log" 2>&1
GATE_B_TAIL=$(tail -1 "$OUT/gate_b.log")
echo "$GATE_B_TAIL"
echo "$GATE_B_TAIL" | grep -qE " failed| error" && FAILED="$FAILED gate_b"

echo "== Gate C: full regression =="
$PY -m pytest tests -q -p no:cacheprovider -rf > "$OUT/gate_c.log" 2>&1
GATE_C_TAIL=$(tail -1 "$OUT/gate_c.log")
echo "$GATE_C_TAIL"
GATE_C_VERDICT=$($PY - "$OUT/gate_c.log" "$BASELINE" <<'PYEOF'
import json, sys

def norm(s):
    s = s.strip()
    for p in ("FAILED ", "ERROR "):
        if s.startswith(p):
            s = s[len(p):]
    return s.split(" - ")[0].strip()

log, baseline_path = sys.argv[1], sys.argv[2]
names = {norm(l) for l in open(log, errors="replace") if l.startswith(("FAILED ", "ERROR "))}
names.discard("")
try:
    baseline = {norm(l) for l in json.load(open(baseline_path))["known_environmental"]}
    baseline.discard("")
except Exception:
    baseline = set()
unexpected = sorted(names - baseline)
missing = sorted(baseline - names)
print(json.dumps({"failures": len(names), "unexpected": unexpected, "baseline_absent": missing}))
PYEOF
)
echo "$GATE_C_VERDICT"
echo "$GATE_C_VERDICT" | grep -q '"unexpected": \[\]' || FAILED="$FAILED gate_c"

echo "== Gate D: static =="
GATE_D="ok"
export GATE_D
$PY -m ruff check src tests > /dev/null 2>&1 || { GATE_D="ruff FAILED"; FAILED="$FAILED gate_d_ruff"; }
$PY -m mypy src/rosclaw > /dev/null 2>&1 || { GATE_D="$GATE_D mypy FAILED"; FAILED="$FAILED gate_d_mypy"; }
$PY -m compileall -q src/rosclaw || { GATE_D="$GATE_D compileall FAILED"; FAILED="$FAILED gate_d_compileall"; }

echo "== db status / doctor =="
$PY -m rosclaw.cli db status --json > "$OUT/db_status.json" 2>&1 || true
$PY -m rosclaw.cli db doctor --json > "$OUT/db_doctor.json" 2>&1 || true

RESULT="PASS"
[ -n "$FAILED" ] && RESULT="FAIL"
FAILED="$FAILED" RESULT="$RESULT" $PY - "$REPORT" "$OUT" <<'PYEOF'
import json, os, subprocess, sys

def _from_first_brace(s):
    i = s.find("{")
    return s[i:] if i >= 0 else "{}"

report_path, out = sys.argv[1], sys.argv[2]
report = {
    "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "python": sys.version.split()[0],
    "gate_a": json.loads(_from_first_brace(open(f"{out}/gate_a.json", errors="replace").read())),
    "gate_b_tail": open(f"{out}/gate_b.log", errors="replace").read().strip().splitlines()[-1],
    "gate_c_tail": open(f"{out}/gate_c.log", errors="replace").read().strip().splitlines()[-1],
    "gate_d": os.environ.get("GATE_D", "ok"),
    "failed_gates": [g for g in os.environ.get("FAILED", "").split() if g],
    "result": os.environ["RESULT"],
}
open(report_path, "w").write(json.dumps(report, indent=2))
print(json.dumps({"result": report["result"], "report": report_path}))
PYEOF
[ "$RESULT" = "PASS" ]
