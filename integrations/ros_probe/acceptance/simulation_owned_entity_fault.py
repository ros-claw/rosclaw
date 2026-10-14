"""SDK operator removal/recreation of one pinned disposable Gazebo model.

No ROS, Runtime, driver, velocity publisher or goal. Original model XML elements
are extracted unchanged; fresh ECM records must show absence and a different ID.
Removal is explicit: replacement stop evidence never proves old-entity speed.
"""

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from simulation_current_facts import _read
from simulation_witness_sources import SimulationWitnessSources


class ExpiredOwnedEcmObservationError(ValueError):
    """An observation gap is UNKNOWN, never evidence of current identity."""


def latest_owned_ecm(root, policy):
    paths = sorted(
        p
        for p in root.glob("ecm-original.jsonl*")
        if re.fullmatch(r"ecm-original\.jsonl(?:\.[0-9]+)?", p.name)
    )
    if not 1 <= len(paths) <= 8:
        raise ValueError("bounded original owned ECM segments required")
    fd = os.open(paths[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 15_000_000:
            raise ValueError("bounded original ECM segment required")
        raw = os.pread(fd, min(info.st_size, 1_000_000), max(0, info.st_size - 1_000_000))
    finally:
        os.close(fd)
    complete = raw[: raw.rfind(b"\n") + 1]
    if not complete:
        raise ValueError("complete original ECM record required")
    line = complete.splitlines(keepends=True)[-1]
    if len(line) > 500_000:
        raise ValueError("bounded original ECM sample required")
    sample = json.loads(line)
    enrollment = json.loads(_read(root / "execution-facts-enrollment.json", 128000))
    pins = enrollment["source_pins"]
    if (
        sample["observer_run_label"] != policy["run_id"]
        or sample["source_stream"]["observer_epoch"] != pins["observer_epoch"]
        or sample["source_stream"]["observer_pid"] != pins["observer_pid"]
    ):
        raise ValueError("fresh unchanged observer identity/run required")
    age = time.monotonic_ns() - sample["observer_monotonic_ns"]
    if age < 0:
        raise ValueError("future original ECM clock rejected")
    if age > 500_000_000:
        raise ExpiredOwnedEcmObservationError(
            "original ECM expired during model loading; current identity UNKNOWN"
        )
    return {"original_record_sha256": hashlib.sha256(line).hexdigest(), "sample": sample}


def replace_owned_model(root, assets, policy, *, description_refresh=None):
    world_raw = _read(assets / "world.sdf", 20_000_000)
    if hashlib.sha256(world_raw).hexdigest() != policy["world_sha256"]:
        raise ValueError("original pinned World required for model replacement")
    source_raw = _read(assets / "observer-sources.json", 128000)
    sources = SimulationWitnessSources.from_bytes(
        source_raw,
        expected_sha256=hashlib.sha256(source_raw).hexdigest(),
        urdf_bytes=_read(assets / "robot.urdf", 5_000_000),
        world_sdf_bytes=world_raw,
    )
    if sources.run_id != policy["run_id"]:
        raise ValueError("owned replacement run differs from original source")
    world = ET.fromstring(world_raw).find("world")
    robot, world_id = sources.robot_entity, world.get("name")
    models = [m for m in world.findall("model") if m.get("name") == robot]
    if len(models) != 1:
        raise ValueError("one declared original model required")
    before = latest_owned_ecm(root, policy)
    ids = [m["entity"] for m in before["sample"]["models"] if m["name"] == robot]
    if len(ids) != 1:
        raise ValueError("one original live model identity required")
    original_id = ids[0]
    sdf = ET.Element("sdf", version="1.9")
    sdf.append(models[0])
    model_raw = ET.tostring(sdf)
    with (root / "governed-replacement-model-original.sdf").open("xb") as stream:
        stream.write(model_raw)
    operations = []
    stale_unknowns = []

    def gz(service, kind, request):
        started = time.monotonic_ns()
        argv = [
            "gz",
            "service",
            "-s",
            f"/world/{world_id}/{service}",
            "--reqtype",
            kind,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "5000",
            "--req",
            request,
        ]
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
            raise ValueError("actual owned Gazebo model operation not acknowledged")

    def wait_inventory(replaced):
        deadline = time.monotonic_ns() + 5_000_000_000
        while time.monotonic_ns() < deadline:
            try:
                row = latest_owned_ecm(root, policy)
            except ExpiredOwnedEcmObservationError as exc:
                stale_unknowns.append(
                    {"monotonic_ns": time.monotonic_ns(), "unknown_reason": str(exc)}
                )
                time.sleep(0.02)
                continue
            current = [m["entity"] for m in row["sample"]["models"] if m["name"] == robot]
            if (not replaced and not current) or (
                replaced and len(current) == 1 and current[0] != original_id
            ):
                return row
            time.sleep(0.02)
        raise ValueError("original ECM did not confirm model removal/replacement")

    started = time.monotonic_ns()
    gz("remove", "gz.msgs.Entity", f"name: {json.dumps(robot)}, type: MODEL")
    removed = wait_inventory(False)
    description_effect = None
    with tempfile.NamedTemporaryFile(prefix="owned-model-replacement-", suffix=".sdf") as temporary:
        temporary.write(model_raw)
        temporary.flush()
        log_offset = (
            (root / "gazebo.log").stat().st_size if description_refresh is not None else None
        )
        gz(
            "create",
            "gz.msgs.EntityFactory",
            f"sdf_filename: {json.dumps(temporary.name)}, allow_renaming: false",
        )
        if description_refresh is not None:
            description_effect = description_refresh.after_create(log_offset)
        replacement = wait_inventory(True)
    replacement_id = next(
        m["entity"] for m in replacement["sample"]["models"] if m["name"] == robot
    )
    return {
        "world_id": world_id,
        "robot_name": robot,
        "original_entity_id": original_id,
        "replacement_entity_id": replacement_id,
        "original_body_removed": True,
        "replacement_body_reenrolled": False,
        "replacement_model_sha256": hashlib.sha256(model_raw).hexdigest(),
        "before_monotonic_ns": started,
        "after_monotonic_ns": time.monotonic_ns(),
        "original_before_ECM": before,
        "original_removed_ECM": removed,
        "original_replacement_ECM": replacement,
        "actual_original_operations": operations,
        "identical_original_description_refresh": description_effect,
        "loading_observation_unknowns": stale_unknowns,
        "physical_stop_scope": "original entity absent; independent replacement-entity physics after identity loss; never old-entity speed",
    }
