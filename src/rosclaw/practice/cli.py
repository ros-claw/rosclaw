"""Practice CLI - Episode recording and replay."""
import json
from pathlib import Path


def _artifact_dir() -> Path:
    return Path.home() / ".rosclaw" / "artifacts" / "episodes"


def list_episodes():
    artifacts = _artifact_dir()
    if not artifacts.exists():
        print("No episodes recorded yet.")
        print(f"Artifact directory: {artifacts}")
        return
    episodes = []
    for d in sorted(artifacts.glob("ep_*")):
        meta = d / "metadata.json"
        if meta.exists():
            with open(meta, encoding="utf-8") as f:
                data = json.load(f)
            episodes.append({
                "id": d.name,
                "robot": data.get("robot_id", "unknown"),
                "status": data.get("status", "UNKNOWN"),
                "reward": data.get("reward", 0.0),
            })
    if not episodes:
        print("No episodes recorded yet.")
        return
    print(f"{'Episode ID':<15} {'Robot':<15} {'Status':<10} {'Reward':<8}")
    print("-" * 50)
    for ep in episodes:
        print(f"{ep['id']:<15} {ep['robot']:<15} {ep['status']:<10} {ep['reward']:<8.2f}")


def show_episode(episode_id):
    meta_path = _artifact_dir() / episode_id / "metadata.json"
    if not meta_path.exists():
        print(f"Episode {episode_id} not found.")
        print(f"Searched: {meta_path}")
        return
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2, default=str))


def replay_episode(episode_id):
    art = _artifact_dir() / episode_id
    if not art.exists():
        print(f"Episode {episode_id} not found.")
        return
    meta_path = art / "metadata.json"
    traj_path = art / "trajectory.jsonl"
    trace_path = art / "provider_trace.jsonl"
    sandbox_path = art / "sandbox_replay.json"

    print(f"=== Replay: {episode_id} ===")

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Robot: {meta.get('robot_id', 'unknown')}")
        print(f"Status: {meta.get('status', 'UNKNOWN')}")
        print(f"Reward: {meta.get('reward', 0.0)}")
        print(f"Events: {meta.get('received_events', [])}")
        print()

    if traj_path.exists():
        print("--- Trajectory ---")
        with open(traj_path, encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                print(f"  [{event.get('phase', '?')}] {event.get('skill_name', '?')}")
        print()

    if trace_path.exists():
        print("--- Provider Traces ---")
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                print(f"  {event.get('status', '?')}")
        print()

    if sandbox_path.exists():
        print("--- Sandbox Replay ---")
        with open(sandbox_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Blocked: {data.get('blocked', False)}")
        print(f"  Reason: {data.get('block_reason', 'N/A')}")
        print()


def validate_episode(session_dir: Path) -> dict:
    """Validate a practice session directory.

    Checks:
    - events.jsonl exists and is parseable
    - timestamps are monotonically increasing
    - artifact references in payload_ref exist on disk
    """
    events_path = session_dir / "raw" / "events.jsonl"
    result = {
        "valid": False,
        "errors": [],
        "event_count": 0,
        "missing_refs": [],
    }
    if not events_path.exists():
        result["errors"].append(f"events.jsonl not found: {events_path}")
        return result

    last_ts = -1
    with open(events_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                result["errors"].append(f"Line {line_no}: JSON decode error: {e}")
                continue
            result["event_count"] += 1
            ts = event.get("timestamp_ns")
            if ts is not None and last_ts is not None and ts < last_ts:
                result["errors"].append(f"Line {line_no}: timestamp non-monotonic ({ts} < {last_ts})")
            if ts is not None:
                last_ts = ts
            for ref_key, ref_path in event.get("payload_ref", {}).items():
                full = session_dir / ref_path
                if not full.exists():
                    result["missing_refs"].append(f"Line {line_no}: {ref_key} -> {ref_path}")
    if result["missing_refs"]:
        result["errors"].extend(result["missing_refs"])
    result["valid"] = not result["errors"]
    return result


def export_episode(session_dir: Path, output_dir: Path) -> Path:
    """Export a validated session directory to a portable archive."""
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / f"{session_dir.name}.tar.gz"
    shutil.make_archive(str(export_path.with_suffix("")), "gztar", root_dir=str(session_dir))
    return export_path
