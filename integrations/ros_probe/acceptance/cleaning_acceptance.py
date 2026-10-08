"""Supplemental acceptance of a completed live MCP/daemon cleaning journey."""

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from observations import latest_completed_observation

from rosclaw.connectors.ros.verification.mission import verify_mission
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SQLiteStructuredStore


def native_path_sources(root):
    """Legacy snapshots or current append-only streams, never guessed paths."""
    legacy = [root / name for name in ("coverage_path.json", "navigation_path.json")]
    if all(path.is_file() for path in legacy):
        return legacy
    streams = sorted(root.glob("plan-events-*.jsonl"))
    if not streams:
        raise RuntimeError("Native path evidence is missing; no legacy snapshots or append log")
    return streams


def mission_practice_episode(root, mission_id):
    """Find exactly this mission's Practice record; never infer success from a name."""
    sessions = root / "practice/sessions"
    candidates = list(sessions.glob("*/episode.json")) + list(
        sessions.glob("*/episodes/*/episode.json")
    )
    if len(candidates) > 64:
        raise ValueError("isolated mission Practice lookup exceeds bound")
    matching = {}
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_relative_to(sessions.resolve()) or path.stat().st_size > 2_000_000:
            raise ValueError("bounded local Practice record required")
        record = json.loads(path.read_bytes())
        if type(record) is not dict:
            raise ValueError("Practice record must be an object")
        if record.get("practice_id") == mission_id:
            matching[path] = record
    if len(matching) != 1:
        raise ValueError("exactly one Practice record for the evidence mission required")
    episode, record = next(iter(matching.items()))
    if record.get("outcome") != "SUCCESS":
        raise ValueError("evidence mission Practice was not closed successfully")
    manifest = episode.with_name("manifest.yaml")
    if not manifest.is_file() or not manifest.resolve().is_relative_to(sessions.resolve()):
        raise ValueError("same local Practice episode manifest required")
    if manifest.stat().st_size > 2_000_000:
        raise ValueError("bounded local Practice manifest required")
    metadata = yaml.safe_load(manifest.read_bytes())
    if (
        type(metadata) is not dict
        or metadata.get("practice_id") != mission_id
        or type(record.get("session_id")) is not str
        or not record["session_id"]
        or metadata.get("session_id") != record["session_id"]
    ):
        raise ValueError("Practice manifest mission/session identity mismatch")
    return episode, manifest


def main(root, fixture, output, native=False):
    receipt_export = None
    if native:
        receipt_export = root / "canonical-receipts-final.json"
        rows = json.loads(receipt_export.read_text())
        names = {
            "localization.set_initial_pose": "golden-localize",
            "coverage.execute": "golden-coverage",
            "ros.expert.remember": "golden-remember",
        }
        if len(rows) != 3 or {row["capability_id"] for row in rows} != set(names):
            raise RuntimeError("exactly three required Native canonical responses expected")
        receipts = {names[row["capability_id"]]: row for row in rows}
        with sqlite3.connect(f"file:{root / 'home/agentd/missions.db'}?mode=ro", uri=True) as db:
            task = db.execute("select state from tasks order by created_at desc limit 1").fetchone()
            if task != ("SUCCEEDED",):
                raise RuntimeError("Native TaskKernel did not succeed")
        if not json.loads((root / "sdk-usage.json").read_text()):
            raise RuntimeError("actual Native SDK usage required")
    else:
        receipts = {
            name: json.loads((root / f"{name}.receipt.json").read_text())
            for name in ("golden-localize", "golden-coverage", "golden-remember")
        }
    by_action = {row["receipt"]["action_id"]: row for row in receipts.values()}
    if len(by_action) != 3:
        raise RuntimeError("distinct canonical action IDs required")
    if any(
        r["receipt"]["final_state"] != "COMPLETED"
        or r["receipt"]["evidence_domain"] != "SIMULATION"
        or r["receipt"]["usable_for_real_execution"]
        for r in receipts.values()
    ):
        raise RuntimeError("three completed SIM canonical receipts required")
    artifact_path = next(
        p
        for p in (root / "actions").glob("rosevidence_*.json")
        if not p.name.endswith(".verification.json")
    )
    evidence = json.loads(artifact_path.read_text())
    original = json.loads(artifact_path.with_suffix(".verification.json").read_text())

    class SavedCanonicalReceipts:
        def get_execution_receipt(self, action_id):
            return by_action[action_id]

    replay = verify_mission(evidence, daemon=SavedCanonicalReceipts())
    if replay != original or not replay["success"]:
        raise RuntimeError("independent coverage replay or exact receipt binding failed")
    memory = MemoryInterface(
        evidence["body_id"], seekdb_client=SQLiteStructuredStore(str(root / "memory.sqlite"))
    )
    memory.initialize()
    try:
        row = memory.get_experience(evidence["mission_id"])
        canonical = receipts["golden-coverage"]["receipt"]
        expected_duration = (
            datetime.fromisoformat(canonical["finished_at"])
            - datetime.fromisoformat(canonical["started_at"])
        ).total_seconds()
        if (
            row is None
            or row["outcome"] != "success"
            or not math.isclose(row["duration_sec"], expected_duration, abs_tol=1e-6)
            or row["metadata"]["raw"]["verification"] != original
        ):
            raise RuntimeError("live Memory semantic/timing/verification persistence failed")
        queries = {}
        for query in (
            "complete room cleaning",
            "cleaning coverage",
            "区域清扫",
            "完成整个房间清扫。",
            "clean the entire room",
        ):
            rows = memory.find_similar_experiences(query, outcome_filter="success")
            if not rows or rows[0]["id"] != evidence["mission_id"]:
                raise RuntimeError("live accepted cleaning mission not retrievable")
            queries[query] = rows[0]["id"]
    finally:
        memory.stop()
    episode, practice_manifest = mission_practice_episode(root, evidence["mission_id"])
    samples = []
    until = time.monotonic() + 3
    while time.monotonic() < until:
        sample = latest_completed_observation(fixture / "witness.jsonl")
        age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
        if (
            not 0 <= age < 0.3
            or not sample["observation_complete"]
            or sample["cleaning_enabled"]
            or sample["lease_remaining_sec"] > 0
            or sample["collision_count"]
        ):
            raise RuntimeError("fresh independent cleanup/standstill observation failed")
        samples.append(sample)
        time.sleep(0.1)
    displacement = max(
        math.hypot(s["x"] - samples[0]["x"], s["y"] - samples[0]["y"]) for s in samples
    )
    rotation = max(
        abs(
            math.atan2(
                math.sin(s["yaw"] - samples[0]["yaw"]), math.cos(s["yaw"] - samples[0]["yaw"])
            )
        )
        for s in samples
    )
    if displacement > 0.01 or rotation > 0.03:
        raise RuntimeError("post-cleaning independent standstill failed")
    output.mkdir(exist_ok=False, parents=True)
    memory_snapshot = output / "memory.snapshot.sqlite"
    with (
        sqlite3.connect(f"file:{root / 'memory.sqlite'}?mode=ro", uri=True) as source,
        sqlite3.connect(memory_snapshot) as target,
    ):
        source.backup(target)
    files = []

    def archive(source, name, compressed=False, live_prefix=False):
        raw = source.read_bytes()
        if live_prefix:
            # The independent observer is still needed for the standstill
            # check. Capture complete lines only; final EOF/chain validation
            # remains a separate mandatory audit after stack shutdown.
            raw = raw[: raw.rfind(b"\n") + 1]
            if not raw:
                raise RuntimeError("Native path observation prefix is empty")
        target = output / name
        target.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
        files.append(
            {
                "path": name,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "gzip": compressed,
                **({"source_role": "LIVE_PREFIX_NOT_FINAL_AUDIT"} if live_prefix else {}),
            }
        )

    for name, response in receipts.items():
        if native:
            raw = (json.dumps(response, indent=2) + "\n").encode()
            target = output / f"{name}.receipt.json"
            target.write_bytes(raw)
            files.append(
                {
                    "path": target.name,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "archive_sha256": hashlib.sha256(raw).hexdigest(),
                    "gzip": False,
                    "source_kind": "exported saved canonical response; descriptive filename retains original action ID",
                }
            )
        else:
            archive(root / f"{name}.receipt.json", f"{name}.receipt.json")
    if native:
        archive(receipt_export, "canonical-receipts-final.json.gz", True)
        archive(root / "sdk-usage.json", "sdk-usage.json")
        archive(root / "usage.json", "core-usage.json")
    archive(artifact_path, artifact_path.name + ".gz", True)
    archive(
        artifact_path.with_suffix(".verification.json"),
        artifact_path.with_suffix(".verification.json").name,
    )
    archive(episode, "practice-episode.json")
    archive(memory_snapshot, "memory.snapshot.sqlite.gz", True)
    memory_snapshot.unlink()
    archive(practice_manifest, "practice-manifest.yaml")
    for name in (
        "body.json",
        "execution_config.json",
        "measured_map.json",
        "robot.urdf",
        "daemon.log",
    ):
        archive(root / name, name + ".gz", True)
    for name in (
        ("fixture_profile.json", "probe.log")
        if native
        else ("snapshot.json", "task_graph.json", "solution.json")
    ):
        archive(root / name, name + ".gz", True)
    if native:
        for source in native_path_sources(root):
            is_stream = source.suffix == ".jsonl"
            name = source.name + (".live-prefix.gz" if is_stream else ".gz")
            archive(source, name, True, live_prefix=is_stream)
    result = {
        "status": "PASS",
        "source_commit": json.loads((root / "source-freeze.json").read_text())["git_sha"],
        "evidence_domain": "SIMULATION",
        "autonomous_llm": native,
        "task_kernel_succeeded": native,
        "hardware_verified": False,
        "causal_memory_benefit_verified": False,
        "mission_id": evidence["mission_id"],
        "coverage_ratio": replay["coverage"]["coverage_ratio"],
        "collision_count": replay["safety"]["collision_count"],
        "trace_gaps": replay["coverage"]["trace_gaps"],
        "complete_independent_observations": replay["safety"]["observation_complete"],
        "duration_sec": expected_duration,
        "memory_instruction": row["instruction"],
        "memory_duration_sec": row["duration_sec"],
        "retrieval_after_reopen": queries,
        "practice_outcome": "SUCCESS",
        "canonical_receipts": "3 COMPLETED / SIMULATION",
        "post_cleanup_displacement_m": displacement,
        "post_cleanup_yaw_change_rad": rotation,
        "post_cleanup_observations": samples,
        "files": files,
        "verification_replay_uses_saved_canonical_receipts": True,
    }
    (output / "acceptance.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("files", "post_cleanup_observations")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    main(args.root, args.fixture, args.output, native=args.native)
