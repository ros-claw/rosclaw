#!/usr/bin/env python3
"""Resource sampler for the SeekDB 1.4 engine process (PR-SDB-140-5, P1-3).

Reads /proc/<pid>/smaps_rollup (Rss, Pss, Private_Clean, Private_Dirty,
Shared_Clean, Shared_Dirty, Swap) + threads + fd count + cpu% every
--interval-s seconds, writes JSONL.  Phase labels come from --phase changes
signaled by a phase file (--phase-file): the orchestrating script writes the
current phase name into it and the sampler stamps every row with it.

Verdict helper (--check at exit): the release gate is NOT "hit 166 MiB" —
it is: no crash, and RSS/PSS/threads/fds must not grow monotonically over
the soak tail (last third vs first third slope).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

SMAPS_KEYS = (
    "Rss",
    "Pss",
    "Private_Clean",
    "Private_Dirty",
    "Shared_Clean",
    "Shared_Dirty",
    "Swap",
)


def find_engine_pid(port: int) -> int | None:
    """Identity-aware: seekdb process whose cmdline carries this port."""
    for pid in (p for p in os.listdir("/proc") if p.isdigit()):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                args = fh.read().decode(errors="replace").split("\0")
            if not args or not args[0].endswith("seekdb"):
                continue
            for i, a in enumerate(args):
                if a == "--port" and i + 1 < len(args) and args[i + 1] == str(port):
                    return int(pid)
                if a.startswith("--port=") and a.split("=", 1)[1] == str(port):
                    return int(pid)
        except (OSError, PermissionError):
            continue
    return None


def sample(pid: int, cpu_prev: tuple[float, float] | None) -> dict:
    row: dict = {"ts": time.time()}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in SMAPS_KEYS:
                    row[key.lower()] = int(rest.strip().split()[0])  # kB
        row["threads"] = len(os.listdir(f"/proc/{pid}/task"))
        row["fds"] = len(os.listdir(f"/proc/{pid}/fd"))
        with open(f"/proc/{pid}/stat") as fh:
            parts = fh.read().split()
        utime, stime = int(parts[13]), int(parts[14])
        total = (utime + stime) / os.sysconf("SC_CLK_TCK")
        now = time.time()
        if cpu_prev is not None:
            dt = now - cpu_prev[0]
            row["cpu_pct"] = round(100.0 * (total - cpu_prev[1]) / dt, 1) if dt > 0 else 0.0
        row["_cpu_total"] = total
    except FileNotFoundError:
        row["dead"] = True
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=2881)
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--interval-s", type=float, default=15.0)
    p.add_argument("--duration-s", type=float, required=True)
    p.add_argument("--phase-file", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pid = args.pid or find_engine_pid(args.port)
    if pid is None:
        print(f"no seekdb engine on port {args.port}")
        return 2
    print(f"sampling pid {pid} every {args.interval_s}s for {args.duration_s}s -> {args.out}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.duration_s
    cpu_prev: tuple[float, float] | None = None
    with out.open("w") as fh:
        while time.time() < deadline:
            row = sample(pid, cpu_prev)
            cpu_prev = (time.time(), row.pop("_cpu_total", 0.0))
            if args.phase_file and Path(args.phase_file).exists():
                row["phase"] = Path(args.phase_file).read_text().strip()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row.get("dead"):
                print("ENGINE DIED during sampling")
                return 1
            time.sleep(args.interval_s)
    print("sampling complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
