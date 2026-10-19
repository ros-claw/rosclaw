"""Closed source-only Nav2 SDK owner, with no fault-label or motion interface.

Recipes are operator-owned frozen originals. A model can select a registered
read-only probe ID; it cannot supply paths, scripts, parameters or SDK commands.
This owner alone is not a pilot gate or a model provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
    RECONSTRUCTED_SDK_IMAGE,
    SDK_IMAGE,
)
from benchmarks.ros_expert_ablation.rsk_episode_runner import Closed, OriginalRef, _read
from benchmarks.ros_expert_ablation.rsk_nav2_observation_window import (
    capture_owned_unadmitted_window,
)
from integrations.ros_probe.acceptance.simulation_atomic_read import read_bounded_regular


class Nav2ObservationRecipeV1(Closed):
    schema_version: Literal["rosclaw.rsk_nav2_observation_recipe.v1"] = (
        "rosclaw.rsk_nav2_observation_recipe.v1"
    )
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    assets_directory: str = Field(min_length=1, max_length=4096)
    observer_directory: str = Field(min_length=1, max_length=4096)
    assets_inventory: OriginalRef
    observer_inventory: OriginalRef
    source_urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["readonly_preinitialization_observation", "readonly_localized_observation"]
    seed: Literal[101, 202, 303] = 101
    max_new_physics_worlds: Literal[1] = 1
    max_goals: Literal[0] = 0
    max_model_calls: Literal[0] = 0

    @property
    def fixture_image(self):
        # A property deliberately preserves every V1 serialized byte/field.
        return SDK_IMAGE

    @model_validator(mode="before")
    @classmethod
    def integer_limits(cls, value):
        if any(
            name in value and type(value[name]) is not int
            for name in ["seed", "max_new_physics_worlds", "max_goals", "max_model_calls"]
        ):
            raise ValueError("strict integer operator limits required")
        return value


class Nav2ObservationRecipeV2(Nav2ObservationRecipeV1):
    schema_version: Literal["rosclaw.rsk_nav2_observation_recipe.v2"] = (
        "rosclaw.rsk_nav2_observation_recipe.v2"
    )
    sdk_image: Literal[
        "sha256:8cc3fbf1eaa57058f0c1386c061253a55651106bd46b98fb046439f966c9019b"
    ] = RECONSTRUCTED_SDK_IMAGE

    @property
    def fixture_image(self):
        return self.sdk_image


def parse_nav2_observation_recipe(raw):
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("closed Nav2 observation recipe required")
    version = value.get("schema_version", "rosclaw.rsk_nav2_observation_recipe.v1")
    if version == "rosclaw.rsk_nav2_observation_recipe.v1":
        return Nav2ObservationRecipeV1.model_validate(value)
    if version == "rosclaw.rsk_nav2_observation_recipe.v2":
        return Nav2ObservationRecipeV2.model_validate(value)
    raise ValueError("unsupported Nav2 observation SDK recipe version")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def _directory(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("operator directory must be relative to the explicit evidence root")
    path = root / relative
    path.resolve(strict=True).relative_to(root)
    for component in [path, *path.parents]:
        if component == root:
            break
        if component.is_symlink():
            raise ValueError("operator directory cannot traverse symlinks")
    if not path.is_dir():
        raise ValueError("original operator directory missing")
    return path


def _inventory(root, directory, reference):
    """Complete file closure; the inventory itself must live outside the tree."""
    rows = json.loads(_read(root, reference, limit=2_000_000))
    if type(rows) is not dict or not 1 <= len(rows) <= 2000:
        raise ValueError("bounded complete original file inventory required")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("operator inventory cannot include links or special files")
        if path.is_file():
            name = str(path.relative_to(directory))
            actual.add(name)
            if (
                name not in rows
                or type(rows[name]) is not dict
                or set(rows[name]) != {"sha256", "bytes"}
            ):
                raise ValueError("operator file missing from original inventory")
            raw = read_bounded_regular(path, 64_000_000)
            if (
                type(rows[name]["bytes"]) is not int
                or len(raw) != rows[name]["bytes"]
                or _sha(raw) != rows[name]["sha256"]
            ):
                raise ValueError("operator original inventory SHA/size changed")
    if actual != set(rows):
        raise ValueError("operator original file closure changed")
    return rows


class Nav2SourceOnlyDiagnosticOperator:
    """One owned fixed-image physics World; no callback or user script injection."""

    def __init__(self, *, recipe: OriginalRef, worktree, evidence_root, output_parent):
        self.root = Path(evidence_root).resolve(strict=True)
        self.worktree = Path(worktree).resolve(strict=True)
        self.output_parent = Path(output_parent).resolve(strict=True)
        self.output_parent.relative_to(self.root)
        self.recipe_reference = OriginalRef.model_validate(recipe)
        self.recipe = parse_nav2_observation_recipe(_read(self.root, self.recipe_reference))
        self.calls = 0
        self.last_world = None
        self.started_containers = 0
        self.check_ready()

    def _git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.worktree, text=True, timeout=15
        ).strip()

    def check_ready(self):
        if Path(__file__).resolve().parents[2] != self.worktree:
            raise ValueError("closed SDK owner must run from its exact declared worktree")
        if self.output_parent.resolve(strict=True) != self.output_parent:
            raise ValueError("owned SDK output parent changed to a symlink")
        self.output_parent.relative_to(self.root)
        if self._git("rev-parse", "HEAD") != self.recipe.source_revision or self._git(
            "status", "--porcelain"
        ):
            raise ValueError("exact clean frozen SDK owner source required")
        if parse_nav2_observation_recipe(_read(self.root, self.recipe_reference)) != self.recipe:
            raise ValueError("frozen owner recipe changed")
        self.assets = _directory(self.root, self.recipe.assets_directory)
        self.observer = _directory(self.root, self.recipe.observer_directory)
        _inventory(self.root, self.assets, self.recipe.assets_inventory)
        _inventory(self.root, self.observer, self.recipe.observer_inventory)
        if (
            _sha(read_bounded_regular(self.assets / "robot.urdf", 8_000_000))
            != self.recipe.source_urdf_sha256
        ):
            raise ValueError("original operator URDF differs from frozen recipe")
        if os.getuid() == 0:
            raise ValueError("non-root SDK owner required")

    def collect(self, plan, selected, *, remaining_sec):
        if (
            self.calls
            or selected not in plan.probes
            or selected.request.provider != "EXISTING_NATIVE_READ_ONLY_PROBE"
            or selected.request.operation
            not in {"ENDPOINT_QOS", "CLOCK_TF_WINDOW", "NATIVE_DIAGNOSTICS"}
            or type(remaining_sec) not in (int, float)
            or not math.isfinite(remaining_sec)
            or not 60 <= remaining_sec <= 300
        ):
            raise ValueError(
                "one registered read-only probe and sufficient real host budget required"
            )
        self.check_ready()
        self.calls += 1
        started = time.monotonic()
        run = uuid.uuid4().hex
        out = self.output_parent / ("RSK-nav2-observation-" + run)
        out.mkdir(exist_ok=False)
        self.last_world = out
        _write(
            out / "original-dispatch-intent.json",
            {
                "at": datetime.now(UTC).isoformat(),
                "recipe": self.recipe_reference.model_dump(mode="json"),
                "source": self.recipe.source_revision,
                "run_id": run,
                "plan": plan.model_dump(mode="json"),
                "selected_probe_id": selected.request.id,
                "execution_authority": "NONE",
                "max_goals": 0,
            },
        )
        try:
            assets, execution = out / "assets", out / "execution"
            shutil.copytree(self.assets, assets)
            execution.mkdir()
            self._prepare_assets(assets, run)
            rows = {
                str(p.relative_to(assets)): _sha(p.read_bytes())
                for p in assets.rglob("*")
                if p.is_file() and p.name != "prepared-assets.json"
            }
            (assets / "prepared-assets.json").write_text(
                json.dumps(
                    {"schema_version": "rosclaw.prepared_sim_assets.v1", "file_sha256": rows},
                    indent=2,
                )
            )
            policy = json.loads(read_bounded_regular(assets / "navigation-policy.json", 64_000))
            settings = {
                "generic_source_observer": True,
                "inventory_sha256": _sha((assets / "prepared-assets.json").read_bytes()),
                "run_id": run,
                "fault": "none",
                "live_continuity_hold_seconds": 120,
                "live_dual_supervision": True,
                "seed": self.recipe.seed,
                "observed_body_id": policy["body_id"],
                "navigation_fixture_policy": policy,
                "probe_snapshot_period_sec": 0.1,
                "governed_sink": {
                    "schema_version": "rosclaw.governed_sink_bootstrap.v1",
                    "P03_review_sha256": "5bf3d45a8d862873643ef58c9e5cb6891f282c5327adf8c0fa0aaa8db15b3382",
                },
                "engineering_capture": {
                    "schema_version": "rosclaw.engineering_failure_capture.v1",
                    "case": self.recipe.role,
                    "phase": "before_repair",
                    "run_id": run,
                },
            }
            if (assets / "costmap-witness.json").is_file():
                settings["costmap_consumer_capture"] = {
                    "schema_version": "rosclaw.costmap_consumer_capture.v1",
                    "configuration_sha256": _sha((assets / "costmap-witness.json").read_bytes()),
                }
            _write(out / "settings.json", settings)
            common_git = Path(self._git("rev-parse", "--git-common-dir"))
            if not common_git.is_absolute():
                common_git = (self.worktree / common_git).resolve(strict=True)
            name = "reh-rsk-nav2-observation-" + run[:16]
            argv = [
                "docker",
                "create",
                "--pull",
                "never",
                "--name",
                name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--memory",
                "10g",
                "--cpus",
                "6",
                "--pids-limit",
                "512",
                "--shm-size",
                "512m",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,nodev,size=1g",
            ]
            for value in [
                "HOME=/tmp",
                "ROS_LOG_DIR=/tmp/ros-log",
                "LIBGL_ALWAYS_SOFTWARE=1",
                "__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json",
                "GIT_OPTIONAL_LOCKS=0",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                "ROS_DOMAIN_ID=101",
                "ROS_LOCALHOST_ONLY=1",
                "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
            ]:
                argv += ["--env", value]
            for source, target, writable in [
                (self.worktree, "/repo", False),
                (common_git, str(common_git), False),
                (assets, "/assets", False),
                (self.observer, "/observer", False),
                (execution, "/evidence", True),
                (out / "settings.json", "/settings.json", False),
            ]:
                argv += [
                    "--mount",
                    f"type=bind,src={source},dst={target}" + ("" if writable else ",readonly"),
                ]
            argv += [
                "--entrypoint",
                "bash",
                self.recipe.fixture_image,
                "-c",
                "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
                "export PYTHONPATH=/repo/src:/repo/integrations/ros_probe/acceptance:"
                "$PYTHONPATH && python3 /repo/integrations/ros_probe/acceptance/source_binding_fixture.py",
            ]
            _write(
                out / "preregistration.json",
                {
                    "source": self.recipe.source_revision,
                    "run_id": run,
                    "recipe": self.recipe_reference.model_dump(mode="json"),
                    "argv": argv,
                    "actual_new_physics_Worlds_max": 1,
                    "navigation_goals": 0,
                    "fault_label_required": False,
                },
            )
        except Exception as error:
            _write(
                out / "original-failure.json",
                {
                    "phase": "PRE_CREATE_SETUP",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "automatic_retry": False,
                    "reserved_world_slots": 1,
                    "created_containers": 0,
                },
            )
            raise
        cid, process = None, None
        try:
            cid = subprocess.check_output(argv, text=True, timeout=15).strip()
            self.started_containers += 1
            (out / "container-id.txt").write_text(cid + "\n")
            _write(out / "container-before-start.json", self._inspect(cid))
            with (out / "container-original.log").open("xb") as log:
                process = subprocess.Popen(
                    ["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT
                )
                while not (execution / "engineering-capture-ready.json").is_file():
                    if process.poll() is not None:
                        raise ValueError("owned source-only SDK exited before original capture")
                    if time.monotonic() - started > remaining_sec - 25:
                        raise TimeoutError("bounded original SDK startup exhausted; no retry")
                    time.sleep(0.05)
                capture_owned_unadmitted_window(
                    execution,
                    out / "read-only-window",
                    run_id=run,
                    body_snapshot_hash=None,
                    source_revision=self.recipe.source_revision,
                    operation=selected.request.operation,
                )
                _write(
                    execution / "host_capture_ack.json",
                    {"original_readonly_window_preserved": True},
                )
                if (
                    process.wait(
                        timeout=max(0.001, remaining_sec - (time.monotonic() - started) - 3)
                    )
                    != 0
                ):
                    raise ValueError("owned source-only SDK failed; preserve originals, no retry")
            self.check_ready()
            if time.monotonic() - started > remaining_sec:
                raise TimeoutError("owned SDK exceeded the frozen host budget")
        except Exception as error:
            _write(
                out / "original-failure.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "automatic_retry": False,
                    "reserved_world_slots": 1,
                    "created_containers": self.started_containers,
                },
            )
            raise
        finally:
            if cid is None:
                # A timed-out docker-create may still have created our unique
                # named container. Inspect only that declared owned identity.
                try:
                    state = self._inspect(name)
                except subprocess.CalledProcessError:
                    state = None
                if state is not None:
                    if state["Name"] != "/" + name or state["Image"] != self.recipe.fixture_image:
                        raise ValueError("ambiguous owned create; no unrelated container cleanup")
                    mounts = {m["Destination"]: m for m in state["Mounts"]}
                    if mounts.get("/evidence", {}).get("Source") != str(execution):
                        raise ValueError("ambiguous owned evidence mount; no unrelated cleanup")
                    cid = state["Id"]
                    self.started_containers = 1
                    _write(out / "create-timeout-owned-container.json", state)
            if cid is not None:
                state = self._inspect(cid)
                if (
                    state["Id"] != cid
                    or state["Name"] != "/" + name
                    or state["Image"] != self.recipe.fixture_image
                ):
                    raise ValueError("owned SDK identity changed; no unrelated container cleanup")
                if state["State"]["Running"]:
                    subprocess.run(
                        ["docker", "stop", "--timeout", "3", cid],
                        check=True,
                        capture_output=True,
                        timeout=15,
                    )
                    state = self._inspect(cid)
                _write(out / "container-after-stop.json", state)
                if state["State"]["Running"] or state["State"]["Pid"] != 0:
                    raise ValueError("owned SDK has not stopped; retain original failure")
            if process is not None:
                process.wait(timeout=15)
        return out

    @staticmethod
    def _inspect(cid):
        return json.loads(
            subprocess.check_output(["docker", "inspect", cid], text=True, timeout=15)
        )[0]

    def _prepare_assets(self, assets, run):
        tree = ET.parse(assets / "world.sdf")
        world = tree.getroot().find("world")
        sources = json.loads((assets / "observer-sources.json").read_text())
        for plugin in world.findall("plugin"):
            if plugin.get("name") == "rosclaw::SourceWitness":
                plugin.find("run_id").text = run
                for key, value in [
                    ("segment_max_samples", "200"),
                    ("segment_max_bytes", "15000000"),
                    ("max_segments", "8"),
                ]:
                    element = plugin.find(key)
                    if element is None:
                        element = ET.SubElement(plugin, key)
                    element.text = value
        pose = {101: (0.0, 0.0, 0.0), 202: (0.12, -0.12, 0.2), 303: (-0.12, 0.12, -0.2)}[
            self.recipe.seed
        ]
        element = world.find("model[@name='" + sources["robot_entity"] + "']/pose")
        values = [float(value) for value in element.text.split()]
        values[0], values[1], values[5] = pose
        element.text = " ".join(map(str, values))
        tree.write(assets / "world.sdf")
        for name in ["source-initialization.json", "source-preparation.json"]:
            document = json.loads((assets / name).read_text())
            document["run_id"] = run
            document.pop("operator_test_case", None)
            if name == "source-initialization.json":
                document["pose"].update(x=pose[0], y=pose[1], yaw=pose[2])
            (assets / name).write_text(json.dumps(document, indent=2))
        (assets / "run_id.txt").write_text(run + "\n")
        sources.update(run_id=run, world_sdf_sha256=_sha((assets / "world.sdf").read_bytes()))
        (assets / "observer-sources.json").write_text(json.dumps(sources, indent=2))
        refresh = assets / "localization-refresh.json"
        if self.recipe.role == "readonly_localized_observation":
            namespace = sources["observer_node"].rsplit("/", 1)[0]
            refresh.write_text(
                json.dumps(
                    {
                        "schema_version": "rosclaw.sim_localization_refresh.v1",
                        "run_id": run,
                        "service": namespace + "/request_nomotion_update",
                        "period_sec": 0.1,
                        "operator_owned_disposable_sim_only": True,
                    },
                    indent=2,
                )
            )
        elif refresh.exists():
            refresh.unlink()
