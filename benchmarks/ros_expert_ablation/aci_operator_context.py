"""Read original operator artifacts; never opens ROS/DDS or issues actions."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.sdk_operator import SDK_IMAGE
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.contracts.common import content_hash

EXTRA_FIXTURE_SHA256 = "2e47d353693fd42c0c982d047bcec70284fa929931ba687e07e0e835991f3eab"
EXTRA_FIXTURE_PATH = "integrations/ros_probe/acceptance/simulation_engineering_endpoint_fixture.py"


def ownership_review(before, case, tools, *, worktree):
    """Historical SDK ownership of one declared staging flag, not live authority."""
    before = Path(before)
    e = before / "execution"
    receipt = e / "engineering-extra-endpoint-original.json"
    if not receipt.exists():
        return {}
    original = json.loads(receipt.read_text())
    settings = json.loads((before / "settings.json").read_text())
    processes = json.loads((e / "owned-stack-processes.json").read_text())
    registration = json.loads((before / "preregistration.json").read_text())
    container = json.loads((before / "container-before-start.json").read_text())
    code = subprocess.check_output(
        ["git", "show", registration["source"] + ":" + EXTRA_FIXTURE_PATH], cwd=worktree
    )
    process = processes.get("engineering_extra_endpoint", {})
    policy = settings.get("engineering_capture", {})
    if (
        hashlib.sha256(code).hexdigest() != EXTRA_FIXTURE_SHA256
        or original.get("run_id") != settings["run_id"]
        or policy.get("run_id") != settings["run_id"]
        or policy.get("case") != "duplicate_velocity_route"
        or policy.get("phase") not in {"before_repair", "rollback"}
        or original.get("owner_pid") != process.get("pid")
        or process.get("argv") != ["python3", "/repo/" + EXTRA_FIXTURE_PATH]
        or original.get("endpoint")
        != tools.staging["bringup.json"].get("additional_command_endpoint")
        or container["Config"]["Image"] != SDK_IMAGE
        or container["HostConfig"]["NetworkMode"] != "none"
        or container["HostConfig"]["Privileged"]
        or not container["HostConfig"]["ReadonlyRootfs"]
        or not any(m.get("Destination") == "/repo" and not m.get("RW") for m in container["Mounts"])
    ):
        raise ValueError("original SDK ownership evidence does not match closed staged role")
    refs = [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in [
            receipt,
            e / "owned-stack-processes.json",
            before / "settings.json",
            before / "preregistration.json",
            before / "container-before-start.json",
        ]
    ]
    review = {
        "case_hash": case.case_hash,
        "staging_hash": content_hash("acistaging", tools.staging),
        "scope": "SDK_OWNED_ADDITIONAL_STAGING_ENDPOINT",
        "declaration": original["endpoint"],
        "node_name": original["namespace"] + "/rosclaw_engineering_extra_endpoint",
        "original_evidence_refs": refs,
        "fixture_source_sha256": EXTRA_FIXTURE_SHA256,
        "live_process_action_authorized": False,
    }
    return {content_hash("aciowner", review): review}


def aci_from_capture(before, *, worktree, evaluator=None, operator_source=None, arm="B"):
    before = Path(before)
    e = before / "execution"
    model = RosSystemModel.from_dict(
        json.loads((e / "engineering-original-system.json").read_text())
    )
    requirements = json.loads((e / "engineering-original-requirements.json").read_text())
    clock = json.loads((e / "engineering-original-candidate-review.json").read_text())[
        "reviewed_at"
    ]
    files = {
        name: (before / "assets" / name).read_text()
        for name in ["robot.urdf", "nav2.yaml", "bringup.json"]
    }
    files.update(
        {
            "native.json": (e / "engineering-original-native.json").read_text(),
            "requirements.json": json.dumps(requirements),
        }
    )
    case = compile_engineering_case(
        model,
        files["robot.urdf"].encode(),
        requirements,
        source_ref=str(e / "engineering-original-system.json"),
        now=datetime.fromisoformat(clock),
        evidence_domain="SEALED_REPLAY",
    )
    radius = case.derived_documents["navigation_observation_review"]["candidate"]["geometry"][
        "physical_radius_m"
    ]
    tools = TaskTools(files, arm=arm, reviewed_at=clock, minimum_radius_m=radius)
    owned = ownership_review(before, case, tools, worktree=worktree)
    original_run = json.loads((before / "settings.json").read_text())["run_id"]
    return PhysicalACI(
        case,
        model,
        tools,
        evaluator=evaluator,
        ownership_reviews=owned,
        operator_source=operator_source,
        original_run_id=original_run,
    )
