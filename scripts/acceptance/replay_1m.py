#!/usr/bin/env python3
"""PR-DB-1 acceptance: 1M-event replay through recorder + catalog.

Generates N synthetic events through PracticeRecorder, finalizes with the
flush barrier, then reconciles generated == JSONL == catalog == event_index.
Writes machine-readable results to the report directory given as argv[1].
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent
from rosclaw.storage.cli import reconcile_practice


def main() -> int:
    report_dir = Path(sys.argv[1])
    report_dir.mkdir(parents=True, exist_ok=True)
    total = int(os.environ.get("REPLAY_EVENTS", "1000000"))
    data_root = report_dir / "replay_data"

    bus = RuntimeBus()
    recorder = PracticeRecorder(bus, data_root=str(data_root), publish_to_event_bus=False)
    recorder.initialize()
    recorder.start()

    practice_id = "prac_replay_1m"
    started = time.monotonic()
    recorder.on_event(
        RuntimeEvent(
            id="start-1",
            timestamp=datetime.datetime.now(datetime.UTC),
            source="replay",
            robot="replay_bot",
            type="practice.start",
            payload={"practice_id": practice_id, "robot_id": "replay_bot"},
        )
    )

    produce_start = time.monotonic()
    for i in range(total):
        recorder.on_event(
            RuntimeEvent(
                id=f"replay-{i}",
                timestamp=datetime.datetime.now(datetime.UTC),
                source="replay",
                robot="replay_bot",
                type="skill.invoke",
                payload={"seq": i, "kind": "replay"},
            )
        )
    produce_s = time.monotonic() - produce_start

    barrier_start = time.monotonic()
    recorder.on_event(
        RuntimeEvent(
            id="stop-1",
            timestamp=datetime.datetime.now(datetime.UTC),
            source="replay",
            robot="replay_bot",
            type="practice.stop",
            payload={"outcome": "SUCCESS"},
        )
    )
    barrier_s = time.monotonic() - barrier_start
    recorder.stop()

    reconcile = reconcile_practice(practice_id, str(data_root))
    result = {
        "generated_events": total,
        "produce_seconds": round(produce_s, 2),
        "events_per_second": round(total / produce_s, 1),
        "finalize_barrier_seconds": round(barrier_s, 2),
        "total_seconds": round(time.monotonic() - started, 2),
        "flush_barrier": recorder.last_flush_barrier,
        "reconcile": reconcile,
        "passed": bool(reconcile["passed"]),
    }
    (report_dir / "replay_1m.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "reconcile"}, indent=2))
    print("reconcile passed:", reconcile["passed"], "raw_jsonl:", reconcile["raw_jsonl"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
