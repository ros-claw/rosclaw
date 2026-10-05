"""Check explicit CUDA activity in an exported multiprocess Nsight report.

Unified-memory/page-migration tracing is deliberately a separate gate. An
unsupported driver diagnostic prevents claiming a complete copy trace.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def run(database: Path, validation: Path, output: Path):
    result = json.loads(validation.read_text())
    if result["status"] != "PASS" or not result["content_and_backend_validated"]:
        raise ValueError("actual validated Isaac Resize execution required")
    expected = result["published_target"] + result["warmup_validated_frames"]
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        copies = [
            dict(row)
            for row in connection.execute(
                "SELECT p.pid,p.name,e.label AS copy_kind,count(*) AS operations,sum(bytes) AS bytes "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN PROCESSES p USING(globalPid) "
                "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY m.globalPid,m.copyKind"
            )
        ]
        kernels = [
            dict(row)
            for row in connection.execute(
                "SELECT p.pid,p.name,count(*) AS kernels FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN PROCESSES p USING(globalPid) GROUP BY k.globalPid"
            )
        ]
        apis = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE 'cuMem%' "
                "OR s.value LIKE 'cudaMemcpy%' GROUP BY s.value"
            )
        }
        diagnostics = [
            dict(row)
            for row in connection.execute(
                "SELECT severity,text FROM DIAGNOSTIC_EVENT WHERE severity>1"
            )
        ]
    if len(kernels) != 1 or kernels[0]["kernels"] != expected:
        raise RuntimeError("every validated frame requires an observed real Resize kernel")
    if any(
        word in d["text"].lower()
        for d in diagnostics
        for word in ("dropped", "overflow", "lost events")
    ):
        raise RuntimeError("profiler reported incomplete explicit activity")
    dtoh = [r for r in copies if r["copy_kind"] == "Device-to-Host"]
    htoh = [r for r in copies if r["copy_kind"] == "Host-to-Host"]
    if (
        len(dtoh) != 1
        or dtoh[0]["operations"] != expected
        or dtoh[0]["bytes"] != expected * 1920 * 1080 * 3
        or len(htoh) != 1
        or htoh[0]["operations"] != expected
        or htoh[0]["bytes"] != expected * 480 * 270 * 3
        or len({kernels[0]["pid"], dtoh[0]["pid"], htoh[0]["pid"]}) != 3
        or apis.get("cuMemImportFromShareableHandle", 0) == 0
        or apis.get("cuMemExportToShareableHandle", 0) == 0
    ):
        raise RuntimeError("three-process explicit-copy/kernel/IPC-handle gate failed")
    unified_unsupported = any(
        "Unified Memory trace is not supported" in d["text"] for d in diagnostics
    )
    record = {
        "status": "PARTIAL",
        "explicit_cuda_activity_gate": "PASS",
        "complete_copy_trace_verified": False,
        "zero_copy_verified": False,
        "unified_memory_trace_unsupported": unified_unsupported,
        "evidence_domain": "GPU_FIXTURE",
        "hardware_motion": False,
        "validated_frames_including_warmup": expected,
        "copies": copies,
        "kernels": kernels,
        "cuda_apis": apis,
        "diagnostics": diagnostics,
        "nsight_sqlite_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "note": "Explicit CUDA activity covers three component processes; CPU fallback, validation copies and unsupported Unified Memory tracing prevent a complete zero-copy claim.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"explicit_cuda_activity_gate": "PASS", "complete_copy_trace_verified": False})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("database", "validation", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.database, args.validation, args.output)
