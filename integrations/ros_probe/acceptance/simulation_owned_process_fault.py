"""SDK process faults restricted to declared sensor/rosbridge children in this World."""

import json
import os
import signal
from pathlib import Path

from simulation_current_facts import _read


def pause_owned_process(root, label):
    if label not in {"scan_bridge", "rosbridge"}:
        raise ValueError("only explicitly declared nonphysics bridge faults supported")
    owned = json.loads(_read(Path(root) / "owned-stack-processes.json", 32_000))[label]
    pid = owned["pid"]
    proc = Path("/proc")
    if type(pid) is not int or pid <= 1:
        raise ValueError("original owned bridge process required")
    cmdline = (proc / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if " ".join(owned["argv"]) not in cmdline or (proc / str(pid)).stat().st_uid != os.getuid():
        raise ValueError("owned original bridge argv or UID changed")
    descendants = {pid}
    while True:
        previous = set(descendants)
        for path in proc.glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if int(fields[1]) in descendants:
                    descendants.add(int(path.parent.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        if len(descendants) > 64:
            raise ValueError("bounded original bridge descendant budget exceeded")
        if descendants == previous:
            break
    records = []
    for target in sorted(descendants, reverse=True):
        original = (proc / str(target) / "stat").read_text()
        record = {
            "pid": target,
            "start_time": original.rsplit(")", 1)[1].split()[19],
            "original_stat": original,
            "cmdline": (proc / str(target) / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
        }
        if (proc / str(target)).stat().st_uid != os.getuid():
            raise ValueError("owned descendant UID changed")
        records.append(record)
    for record in records:
        os.kill(record["pid"], signal.SIGSTOP)
    return {
        "label": label,
        "owned_root": owned,
        "paused_original_processes": records,
        "signal": "SIGSTOP",
    }


def resume_owned_process(record):
    for original in reversed(record["paused_original_processes"]):
        path = Path("/proc") / str(original["pid"]) / "stat"
        if path.read_text().rsplit(")", 1)[1].split()[19] != original["start_time"]:
            raise ValueError(
                "original suspended process identity changed; never signal replacement"
            )
    for original in reversed(record["paused_original_processes"]):
        os.kill(original["pid"], signal.SIGCONT)
    return {
        "label": record["label"],
        "signal": "SIGCONT",
        "original_pids": [x["pid"] for x in record["paused_original_processes"]],
    }
