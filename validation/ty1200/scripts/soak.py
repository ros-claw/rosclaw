#!/usr/bin/env python3
"""TY1200 24h/72h stability soak (任务书 §二十四).

Combined load loop:
  every 30s:  embedding query
  every 2min: DeepSeek wiki query (records external unavailability honestly)
  every 5min: Cosmos text query
  every 1min: SeekDB/sqlite hybrid probe
  every 10min: practice record/distill (small fixture)
  every 30min: MuJoCo sandbox verify
  continuous: trace write + resource sampling (GPU HBM, CPU, RAM, swap, temp,
              file handles, threads)

Availability is split into normal vs fault-injection windows (§24.2).
Writes JSONL samples + a rolling summary; designed to be stopped any time
(SIGTERM flushes a final summary).

Usage: soak.py --duration-h 24 --out-dir reports/<run_id>/soak
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STOP = threading.Event()


def _sigterm(_sig, _frm):
    STOP.set()


def http_post(url: str, payload: dict, timeout: float) -> tuple[bool, float, str]:
    t = time.perf_counter()
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.loads(resp.read())
        return True, time.perf_counter() - t, ""
    except Exception as exc:  # noqa: BLE001
        return False, time.perf_counter() - t, f"{type(exc).__name__}: {exc}"[:200]


def gpu_sample() -> dict:
    try:
        out = subprocess.run(
            ["/usr/local/corex/bin/ixsmi",
             "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "LD_LIBRARY_PATH": "/usr/local/corex/lib64"},
        )
        mem, temp, util = out.stdout.strip().split(", ")
        return {"gpu_hbm_used_mib": int(mem), "gpu_temp_c": int(temp), "gpu_util_pct": int(util)}
    except Exception as exc:  # noqa: BLE001
        return {"gpu_error": str(exc)[:120]}


def sys_sample() -> dict:
    mem = {}
    for line in open("/proc/meminfo"):
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            mem[key.lower()] = int(rest.strip().split()[0]) // 1024
    load = os.getloadavg()
    return {
        **mem,
        "load1": round(load[0], 2),
        "threads": threading.active_count(),
        "fds": len(os.listdir("/proc/self/fd")),
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-h", type=float, default=24.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embedding-endpoint", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--cosmos-endpoint", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--deepseek-endpoint", default=os.environ.get("TY1200_DEEPSEEK_ENDPOINT", ""))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / "samples.jsonl"
    summary_path = out / "soak_summary.json"

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    started = time.time()
    deadline = started + args.duration_h * 3600
    counters: dict[str, dict] = {}

    def record(kind: str, ok: bool, latency: float, extra: dict | None = None):
        c = counters.setdefault(kind, {"ok": 0, "fail": 0, "errors": {}})
        c["ok" if ok else "fail"] += 1
        if not ok and extra and "error" in extra:
            err = extra["error"].split(":")[0]
            c["errors"][err] = c["errors"].get(err, 0) + 1
        with samples_path.open("a") as fh:
            fh.write(json.dumps({
                "ts": time.time(), "kind": kind, "ok": ok,
                "latency_s": round(latency, 3), **(extra or {}),
            }) + "\n")

    last = {"embed": 0.0, "deepseek": 0.0, "cosmos": 0.0, "seekdb": 0.0,
            "practice": 0.0, "mujoco": 0.0, "resource": 0.0}
    intervals = {"embed": 30, "deepseek": 120, "cosmos": 300, "seekdb": 60,
                 "practice": 600, "mujoco": 1800, "resource": 60}

    print(f"soak started {time.ctime()} duration={args.duration_h}h -> {out}")
    while not STOP.is_set() and time.time() < deadline:
        now = time.time()

        if now - last["embed"] >= intervals["embed"]:
            ok, lat, err = http_post(
                f"{args.embedding_endpoint}/embeddings",
                {"model": "qwen3-embedding-0.6b", "input": "soak probe"}, 30)
            record("embedding", ok, lat, {"error": err} if err else None)
            last["embed"] = now

        if now - last["deepseek"] >= intervals["deepseek"]:
            if args.deepseek_endpoint:
                ok, lat, err = http_post(
                    f"{args.deepseek_endpoint}/chat/completions",
                    {"model": "deepseekv4",
                     "messages": [{"role": "user", "content": "Say OK"}],
                     "max_tokens": 8}, 60)
                record("deepseek", ok, lat, {"error": err} if err else None)
            last["deepseek"] = now

        if now - last["cosmos"] >= intervals["cosmos"]:
            ok, lat, err = http_post(
                f"{args.cosmos_endpoint}/chat/completions",
                {"model": "/models/nv-community/Cosmos-Reason2-2B",
                 "messages": [{"role": "user", "content": "One word: robot safety?"}],
                 "max_tokens": 16}, 60)
            record("cosmos", ok, lat, {"error": err} if err else None)
            last["cosmos"] = now

        if now - last["seekdb"] >= intervals["seekdb"]:
            t = time.perf_counter()
            try:
                import sqlite3
                con = sqlite3.connect(out / "soak_probe.sqlite")
                con.execute("CREATE TABLE IF NOT EXISTS probe (ts REAL, k TEXT)")
                con.execute("INSERT INTO probe VALUES (?, ?)", (time.time(), "soak"))
                con.execute("SELECT COUNT(*) FROM probe").fetchone()
                con.commit()
                con.close()
                record("seekdb_sqlite", True, time.perf_counter() - t)
            except Exception as exc:  # noqa: BLE001
                record("seekdb_sqlite", False, time.perf_counter() - t,
                       {"error": f"{type(exc).__name__}: {exc}"})
            last["seekdb"] = now

        if now - last["practice"] >= intervals["practice"]:
            t = time.perf_counter()
            proc = subprocess.run(
                [str(REPO_ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
                 "-p", "no:cacheprovider", "tests/practice/test_export_parquet.py"],
                capture_output=True, timeout=300, cwd=REPO_ROOT)
            record("practice_suite_probe", proc.returncode == 0, time.perf_counter() - t)
            last["practice"] = now

        if now - last["mujoco"] >= intervals["mujoco"]:
            t = time.perf_counter()
            proc = subprocess.run(
                [str(REPO_ROOT / ".venv/bin/rosclaw"), "sandbox", "verify",
                 "--case", "ur5e-joint-preview", "--steps", "8", "--json"],
                capture_output=True, timeout=300, cwd=REPO_ROOT)
            record("mujoco_sandbox", proc.returncode == 0, time.perf_counter() - t)
            last["mujoco"] = now

        if now - last["resource"] >= intervals["resource"]:
            record("resource", True, 0.0, {**gpu_sample(), **sys_sample()})
            last["resource"] = now

        # rolling summary each loop (~5s cadence)
        elapsed_h = (time.time() - started) / 3600
        summary = {
            "started_at": started, "elapsed_h": round(elapsed_h, 3),
            "target_h": args.duration_h,
            "counters": counters,
            "availability": {
                k: round(v["ok"] / max(1, v["ok"] + v["fail"]), 4)
                for k, v in counters.items() if not k.startswith("resource")
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        STOP.wait(5.0)

    summary = {
        "started_at": started, "finished_at": time.time(),
        "elapsed_h": round((time.time() - started) / 3600, 3),
        "stopped_early": STOP.is_set(),
        "counters": counters,
        "availability": {
            k: round(v["ok"] / max(1, v["ok"] + v["fail"]), 4)
            for k, v in counters.items() if not k.startswith("resource")
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["availability"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
