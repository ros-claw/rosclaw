"""SDK-owned SIM contact instrumentation trial, with no motion/action admission.

Insert one declared static test collider at an observed chassis surface, then
remove it. Inventory drift intentionally invalidates enrollment. This proves
contact sensitivity, never navigation success or a collision-only stop cause.
"""

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from governed_daemon_sigkill_fixture import last_physics
from simulation_current_facts import _read
from simulation_owned_entity_fault import latest_owned_ecm
from simulation_unadmitted_sink import require_unadmitted_sink
from simulation_witness_sources import SimulationWitnessSources


def recent_original_contacts_clear(root, sources):
    """Cumulative episode counts cannot establish that a contact has ended."""
    seen = set()
    for path in root.glob("contact-messages-*.jsonl"):
        if ".chain." in path.name:
            continue
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 30_000_000:
                raise ValueError("bounded original contact segment required")
            offset = max(0, info.st_size - 1_000_000)
            raw = os.pread(fd, min(info.st_size, 1_000_000), offset)
        finally:
            os.close(fd)
        now_ns = time.monotonic_ns()
        lines = raw[: raw.rfind(b"\n") + 1].splitlines()
        if offset:
            lines = lines[1:]
        for line in lines:
            row = json.loads(line)
            age = now_ns - row["received_monotonic_ns"]
            if age < 0:
                raise ValueError("future original contact rejected")
            if age > 300_000_000:
                continue
            if row["run_id"] != sources.run_id or row["topic"] not in sources.contact_topics:
                raise ValueError("contact run/topic differs from declared original")
            seen.add(row["topic"])
            for contact in row["message"]["contacts"]:
                if all(
                    contact[k]["name"] not in sources.ground_collision_names
                    for k in ["collision1", "collision2"]
                ):
                    return False
    return set(sources.ground_witness_topics).issubset(seen)


def collider_bytes(sample, robot, name):
    if sample.get("inventory_complete") is not True:
        raise ValueError("complete original collider inventory required")
    candidates = []
    for collision in sample["collisions"]:
        if not any(a["name"] == robot and a["model"] for a in collision["ancestry"]):
            continue
        size = ET.fromstring(collision["geometry_sdf"]).find("box/size")
        if size is not None:
            dimensions = [float(x) for x in size.text.split()]
            if len(dimensions) != 3 or not all(math.isfinite(x) and x > 0 for x in dimensions):
                raise ValueError("finite positive observed box required")
            candidates.append((math.prod(dimensions), collision, dimensions))
    if not candidates:
        raise ValueError("observed robot box collision required")
    _, collision, dimensions = max(candidates, key=lambda x: x[0])
    x, y, z, roll, pitch, yaw = collision["world_pose"]
    if not all(math.isfinite(v) for v in [x, y, z, roll, pitch, yaw]):
        raise ValueError("finite original collision pose required")
    if abs(roll) > 0.01 or abs(pitch) > 0.01:
        raise ValueError("upright original box required for declared surface trial")
    x += math.cos(yaw) * dimensions[0] / 2
    y += math.sin(yaw) * dimensions[0] / 2
    sdf = ET.Element("sdf", version="1.9")
    model = ET.SubElement(sdf, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{x} {y} {z} 0 0 {yaw}"
    link = ET.SubElement(model, "link", name="sensitivity_link")
    geometry = ET.SubElement(ET.SubElement(link, "collision", name="sensitivity_box"), "geometry")
    ET.SubElement(ET.SubElement(geometry, "box"), "size").text = "0.12 0.12 0.12"
    return ET.tostring(sdf), collision


def run_trial(root, assets, policy):
    world_raw = _read(assets / "world.sdf", 20_000_000)
    if hashlib.sha256(world_raw).hexdigest() != policy["world_sha256"]:
        raise ValueError("pinned original World required")
    sources_raw = _read(assets / "observer-sources.json", 128000)
    sources = SimulationWitnessSources.from_bytes(
        sources_raw,
        expected_sha256=hashlib.sha256(sources_raw).hexdigest(),
        urdf_bytes=_read(assets / "robot.urdf", 5_000_000),
        world_sdf_bytes=world_raw,
    )
    if sources.run_id != policy["run_id"]:
        raise ValueError("original source run required")
    world = ET.fromstring(world_raw).find("world").get("name")
    before_sink = require_unadmitted_sink(
        root, run_id=policy["run_id"], body_snapshot_hash=policy["body_snapshot_hash"]
    )
    phases = {}
    operations = []

    def observe(label, positive):
        samples = []
        stable_counter = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = last_physics(root)
            age = time.time() - datetime.fromisoformat(row["captured_at"]).timestamp()
            if not 0 <= age <= 0.3:
                time.sleep(0.02)
                continue
            if (
                row["run_id"] != policy["run_id"]
                or row["body_snapshot_hash"] != policy["body_snapshot_hash"]
                or row["lease_enabled"] is not False
            ):
                raise ValueError("same original run/Body with closed lease required")
            counter = row["physics_collision_count"]
            if label == "after_removal_zero" and counter != stable_counter:
                samples = []
                stable_counter = counter
            matches = (
                counter > 0
                if positive
                else (counter == 0 if label == "baseline_zero" else counter == stable_counter)
            )
            if label == "after_removal_zero":
                matches = matches and recent_original_contacts_clear(root, sources)
            if row["observation_complete"] and matches:
                if not samples or row["captured_at"] != samples[-1]["captured_at"]:
                    samples.append(row)
                if (
                    len(samples) >= 3
                    and (
                        datetime.fromisoformat(samples[-1]["captured_at"])
                        - datetime.fromisoformat(samples[0]["captured_at"])
                    ).total_seconds()
                    >= 0.3
                ):
                    phases[label] = samples
                    return
            else:
                samples = []
            time.sleep(0.02)
        raise ValueError("actual original contact phase not independently observed: " + label)

    def gz(service, kind, request):
        argv = [
            "gz",
            "service",
            "-s",
            f"/world/{world}/{service}",
            "--reqtype",
            kind,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "5000",
            "--req",
            request,
        ]
        started = time.monotonic_ns()
        result = subprocess.run(argv, capture_output=True, timeout=8)
        operations.append(
            {
                "argv": argv,
                "before_monotonic_ns": started,
                "after_monotonic_ns": time.monotonic_ns(),
                "returncode": result.returncode,
                "stdout": result.stdout.decode(),
                "stderr": result.stderr.decode(),
            }
        )
        if result.returncode or result.stdout.decode().strip() != "data: true":
            raise ValueError("owned contact collider operation not acknowledged")

    observe("baseline_zero", False)
    before = latest_owned_ecm(root, policy)
    name = "contact_sensitivity_" + policy["run_id"][:12]
    raw, surface = collider_bytes(before["sample"], sources.robot_entity, name)
    with (root / "contact-sensitivity-original.sdf").open("xb") as stream:
        stream.write(raw)
    inserted = False
    try:
        with tempfile.NamedTemporaryFile(
            prefix="owned-contact-sensitivity-", suffix=".sdf"
        ) as temporary:
            temporary.write(raw)
            temporary.flush()
            gz(
                "create",
                "gz.msgs.EntityFactory",
                f"sdf_filename: {json.dumps(temporary.name)}, allow_renaming: false",
            )
            inserted = True
            observe("actual_nonfloor_contact", True)
            positive_ecm = latest_owned_ecm(root, policy)
    finally:
        if inserted:
            gz("remove", "gz.msgs.Entity", f"name: {json.dumps(name)}, type: MODEL")
        # Preserve every original operation and phase even when positive evidence fails.
        (root / "contact-sensitivity-attempt-original.json").write_text(
            json.dumps({"operations": operations, "phases": phases}, indent=2) + "\n"
        )
    observe("after_removal_zero", False)
    after = latest_owned_ecm(root, policy)
    after_sink = require_unadmitted_sink(
        root, run_id=policy["run_id"], body_snapshot_hash=policy["body_snapshot_hash"]
    )
    return {
        "status": "PASS_LOCAL_ACTUAL_CONTACT_SENSITIVITY_NOT_G3_CLOSURE",
        "policy": policy,
        "collider_name": name,
        "original_surface": surface,
        "collider_sha256": hashlib.sha256(raw).hexdigest(),
        "original_before_ECM": before,
        "original_positive_ECM": positive_ecm,
        "original_after_ECM": after,
        "original_operations": operations,
        "original_phases": phases,
        "sink_before": before_sink,
        "sink_after": after_sink,
        "velocity_publishers_created": 0,
        "goals_sent": 0,
        "inventory_drift_expected": True,
        "collision_only_stop_claimed": False,
        "G3_done": False,
        "v1_done": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated SDK SIM role required")
    root, assets = Path("/evidence"), Path("/assets")
    raw = _read(root / "contact-sensitivity-policy.json", 4096)
    if hashlib.sha256(raw).hexdigest() != args.policy_sha256:
        raise ValueError("closed original sensitivity policy SHA required")
    policy = json.loads(raw)
    if set(policy) != {"run_id", "body_snapshot_hash", "world_sha256"}:
        raise ValueError("closed contact sensitivity policy required")
    outcome = run_trial(root, assets, policy)
    with (root / "contact-sensitivity-outcome-original.json").open("x") as stream:
        json.dump(outcome, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
