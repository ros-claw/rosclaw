"""Fresh D2 Native SIM episode through existing Agentd/MCP/daemon acceptance.

The fixture owns scene perturbations and passive observers. This host launcher
opens no robot transport and executes no action implementation. P0 must already
be merged; source/image/plugin/vendor/config are frozen before simulator launch.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from dynamic_bootstrap import prepare_bindings
from dynamic_source_replay import replay_component_occupancy
from independent_stop import collect_stop_geometry
from paired_efficiency import command, wait_ready

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]


def validate_episode_spec(spec):
    keys = {
        "schema_version",
        "case",
        "source_commit",
        "p0_merge_commit",
        "image",
        "image_id",
        "profile",
        "seed",
        "plugin_sha256",
        "vendor_urdf_sha256",
        "mission_timeout_sec",
        "coverage_preset",
        "repair_strategy",
        "target_xy",
        "dwell_sim_sec",
        "introduce_after_cleaning_sim_sec",
        "port",
        "domain",
        "model_provider",
        "model",
    }
    if (
        type(spec) is not dict
        or set(spec) != keys
        or spec["schema_version"] != "rosclaw.dynamic_native_episode.v1"
    ):
        raise ValueError("closed frozen Native episode protocol required")
    if (
        spec["case"] != "D2"
        or type(spec["profile"]) is not str
        or spec["profile"] not in {"waffle", "burger"}
    ):
        raise ValueError("implemented full Native acceptance is D2 on a known SIM fixture")
    for key in ("source_commit", "p0_merge_commit"):
        if type(spec[key]) is not str or not re.fullmatch(r"[0-9a-f]{40}", spec[key]):
            raise ValueError("exact source and prior P0 merge commits required")
    for key in ("plugin_sha256", "vendor_urdf_sha256"):
        if type(spec[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", spec[key]):
            raise ValueError("exact plugin/vendor bytes required")
    if (
        type(spec["image"]) is not str
        or not re.fullmatch(r"[A-Za-z0-9_./:-]{1,256}", spec["image"])
        or type(spec["image_id"]) is not str
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", spec["image_id"])
    ):
        raise ValueError("exact bounded immutable image required")
    if (
        type(spec["coverage_preset"]) is not str
        or spec["coverage_preset"] not in {"perimeter_stateless", "perimeter_stateless_headland"}
        or spec["repair_strategy"] != "pose_aware_robust"
    ):
        raise ValueError("explicit reviewed P0 planning configuration required")
    if any(
        type(spec[k]) is not str or not re.fullmatch(r"[A-Za-z0-9_./:-]{1,256}", spec[k])
        for k in ("model_provider", "model")
    ):
        raise ValueError("explicit frozen actual Native model/provider required")
    for key, low, high in (
        ("seed", 0, 2**31 - 1),
        ("mission_timeout_sec", 60, 1800),
        ("port", 1024, 65535),
        ("domain", 0, 232),
        ("dwell_sim_sec", 10, 30),
        ("introduce_after_cleaning_sim_sec", 1, 120),
    ):
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError("bounded frozen integer episode parameter required: " + key)
    if (
        type(spec["target_xy"]) is not list
        or len(spec["target_xy"]) != 2
        or any(type(v) not in (int, float) or not -1.2 <= v <= 1.2 for v in spec["target_xy"])
    ):
        raise ValueError("bounded frozen scene target required")
    return spec


def run_episode(directory, protocol, plugin, vendor_urdf, native_profile_home):
    with protocol.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("bounded frozen episode protocol required")
    spec = validate_episode_spec(json.loads(raw))
    plugin_raw, urdf_raw = plugin.read_bytes(), vendor_urdf.read_bytes()
    if (
        not 0 < len(plugin_raw) <= 20_000_000
        or not plugin_raw.startswith(b"\x7fELF")
        or not 0 < len(urdf_raw) <= 2_000_000
        or hashlib.sha256(plugin_raw).hexdigest() != spec["plugin_sha256"]
        or hashlib.sha256(urdf_raw).hexdigest() != spec["vendor_urdf_sha256"]
    ):
        raise ValueError("frozen plugin/vendor source bytes differ")
    for name in ("settings.json", "models.json", "auth.json"):
        if not (native_profile_home / "agent" / name).is_file():
            raise ValueError("existing isolated Native model profile required")

    def frozen():
        if (
            protocol.read_bytes() != raw
            or plugin.read_bytes() != plugin_raw
            or vendor_urdf.read_bytes() != urdf_raw
        ):
            raise ValueError("frozen episode inputs changed")
        if command(["git", "rev-parse", "HEAD"], cwd=REPOSITORY) != spec[
            "source_commit"
        ] or command(["git", "status", "--porcelain"], cwd=REPOSITORY):
            raise ValueError("frozen clean episode source required")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", spec["p0_merge_commit"], spec["source_commit"]],
            cwd=REPOSITORY,
            check=True,
        )
        if (
            command(["docker", "image", "inspect", "--format", "{{.Id}}", spec["image"]])
            != spec["image_id"]
        ):
            raise ValueError("frozen episode image changed")
        # The registered prior commit must be the actual P0 PR merge, not an
        # arbitrary ancestor supplied by the host caller.
        merge = json.loads(command(["gh", "api", "repos/ros-claw/rosclaw/pulls/624"]))
        if (
            merge.get("merged") is not True
            or merge.get("merge_commit_sha") != spec["p0_merge_commit"]
        ):
            raise ValueError("actual reviewed P0 PR merge required before dynamic physics")

    frozen()
    live = command(["docker", "ps", "--format", "{{.Names}}"])
    if any(name.startswith(("reh-n02-", "reh-n03-")) for name in live.splitlines()):
        raise ValueError("prior owned physical episode must stop before this launch")
    directory = directory.resolve()
    directory.mkdir(exist_ok=False)
    (directory / "episode-protocol.json").write_bytes(raw)
    mission = f"n03_d2_{spec['profile']}_{spec['seed']}"
    container = f"reh-n03-d2-{spec['profile']}-{spec['seed']}-{time.time_ns()}"
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY / "src")}
    children, logs = [], []
    launched = False
    result = {
        "status": "NOT_VERIFIED",
        "case": "D2",
        "container": container,
        "source_commit": spec["source_commit"],
        "image_id": spec["image_id"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "autonomous_llm": False,
        "task_kernel_succeeded": False,
        "physical_acceptance": "NOT_VERIFIED",
    }

    def start(name, argv):
        stream = (directory / (name + ".log")).open("x")
        logs.append(stream)
        child = subprocess.Popen(
            argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        children.append(child)
        return child

    def step(name, argv, timeout=120):
        frozen()
        with (directory / (name + ".log")).open("x") as stream:
            child = subprocess.Popen(
                argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            try:
                code = child.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f"{name} exited {code}; retained {name}.log")
            finally:
                if child.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)

    try:
        physics = prepare_bindings(
            directory / "bootstrap",
            urdf_path=vendor_urdf,
            library_path=plugin,
            mission_id=mission,
            profile_name=spec["profile"],
        )
        (directory / "physics.json").write_bytes(
            (directory / "bootstrap/physics.json").read_bytes()
        )
        scenario = {
            "schema_version": "rosclaw.dynamic_fixture_scenario.v1",
            "case": "D2",
            "run_id": physics["binding"]["run_id"],
            "mission_id": mission,
            "obstacle_name": physics["binding"]["obstacle_names"][0],
            "target_xy": spec["target_xy"],
            "dwell_sim_sec": spec["dwell_sim_sec"],
            "introduce_after_cleaning_sim_sec": spec["introduce_after_cleaning_sim_sec"],
            "wall_timeout_sec": spec["mission_timeout_sec"] + 120,
        }
        (directory / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
        stack = (
            "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && "
            "python3 /workspace/integrations/ros_probe/acceptance/stack.py --controller-watchdog "
            f"--profile {spec['profile']} --coverage-preset {spec['coverage_preset']} --seed {spec['seed']} "
            "--brush-binding /evidence/bootstrap/brush.json --physics-fixture /evidence/bootstrap/physics.json "
            "--physics-plugin /frozen/passive.so"
        )
        # A timed-out run command can still have created this unique owned container.
        launched = True
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "--gpus",
                "all",
                "-e",
                "NVIDIA_DRIVER_CAPABILITIES=all",
                "-e",
                f"ROS_DOMAIN_ID={spec['domain']}",
                "-e",
                "ROS_LOCALHOST_ONLY=1",
                "-p",
                f"127.0.0.1:{spec['port']}:9090",
                "-v",
                f"{REPOSITORY}:/workspace:ro",
                "-v",
                f"{directory}:/evidence",
                "-v",
                f"{plugin.resolve()}:/frozen/passive.so:ro",
                spec["image"],
                "bash",
                "-c",
                stack,
            ]
        )
        wait_ready(directory, container)
        endpoint = f"ws://127.0.0.1:{spec['port']}"
        step(
            "native-prepare",
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--directory",
                str(directory),
                "--endpoint",
                endpoint,
                "--profile",
                spec["profile"],
                "--mission-id",
                mission,
                "--mission-timeout",
                str(spec["mission_timeout_sec"]),
                "--repair-strategy",
                spec["repair_strategy"],
                "--dynamic-physics",
                "--prepare-only",
            ],
        )
        step(
            "native-configure",
            [
                sys.executable,
                str(ROOT / "configure_native.py"),
                "--directory",
                str(directory),
                "--endpoint",
                endpoint,
            ],
        )
        model_dir = directory / "home/agent"
        model_dir.mkdir(exist_ok=True)
        for name in ("settings.json", "models.json", "auth.json"):
            shutil.copyfile(native_profile_home / "agent" / name, model_dir / name)
            (model_dir / name).chmod(0o600)
        pose = start(
            "independent-pose",
            [
                "docker",
                "exec",
                "-e",
                "PYTHONPATH=/workspace/src",
                container,
                "bash",
                "-c",
                "source /opt/ros/jazzy/setup.bash && python3 /workspace/integrations/ros_probe/acceptance/pose_observer.py "
                f"--output /evidence/independent-pose.jsonl --duration {spec['mission_timeout_sec'] + 120}",
            ],
        )
        scenario_child = start(
            "dynamic-scenario",
            [
                "docker",
                "exec",
                "-e",
                "PYTHONPATH=/workspace/src",
                container,
                "bash",
                "-c",
                "source /opt/ros/jazzy/setup.bash && python3 /workspace/integrations/ros_probe/acceptance/dynamic_scenario.py "
                "--directory /evidence --scenario /evidence/scenario.json",
            ],
        )
        until = time.monotonic() + 10
        while not (directory / "dynamic-scenario-events.jsonl").exists():
            if scenario_child.poll() is not None or time.monotonic() > until:
                raise RuntimeError("owned dynamic scenario did not start")
            time.sleep(0.1)
        step(
            "native-run",
            [
                sys.executable,
                str(ROOT / "native.py"),
                "--directory",
                str(directory),
                "--endpoint",
                endpoint,
                "--required-scenario",
                str(directory / "scenario.json"),
            ],
            timeout=spec["mission_timeout_sec"] + 120,
        )
        scenario_child.wait(timeout=10)
        if scenario_child.returncode or pose.poll() is not None:
            raise RuntimeError("owned scenario/independent pose source failed")
        with (directory / "sdk-usage.json").open("rb") as stream:
            usage_raw = stream.read(2_000_001)
        if len(usage_raw) > 2_000_000:
            raise ValueError("bounded actual Native SDK usage required")
        usage = json.loads(usage_raw)
        if (
            type(usage) is not list
            or not 1 <= len(usage) <= 256
            or any(
                type(row) is not dict
                or row.get("provider") != spec["model_provider"]
                or row.get("model") != spec["model"]
                for row in usage
            )
        ):
            raise ValueError("actual Native SDK model identity differs from frozen protocol")
        result.update(autonomous_llm=True, actual_sdk_turns=len(usage))
        stop = collect_stop_geometry(directory / "independent-pose.jsonl")
        (directory / "independent-stop.json").write_text(json.dumps(stop, indent=2) + "\n")
        step(
            "native-acceptance",
            [
                sys.executable,
                str(ROOT / "cleaning_acceptance.py"),
                "--root",
                str(directory),
                "--fixture",
                str(directory),
                "--output",
                str(directory / "accepted"),
                "--native",
            ],
        )
        result.update(
            live_canonical_acceptance="PASS_REQUIRES_CLOSED_SOURCE",
            task_kernel_succeeded=True,
        )
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="FAIL", error=type(exc).__name__ + ": " + str(exc))
    finally:
        cleanup_errors = []
        try:
            (directory / "stop-dynamic-scenario.json").write_text("{}\n")
        except Exception as exc:
            cleanup_errors.append("scene stop flag: " + str(exc))
        for child in children:
            try:
                if child.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
            except Exception as exc:
                cleanup_errors.append("owned child: " + str(exc))
        if launched:
            try:
                command(["docker", "stop", "-t", "45", container], timeout=60)
                if (
                    command(["docker", "inspect", "--format", "{{.State.Running}}", container])
                    != "false"
                ):
                    raise RuntimeError("owned simulator still running")
            except Exception as exc:
                cleanup_errors.append("owned simulator: " + str(exc))
        for stream in logs:
            try:
                stream.close()
            except Exception as exc:
                cleanup_errors.append("owned log: " + str(exc))
        if cleanup_errors:
            result.update(status="FAIL", cleanup_errors=cleanup_errors)
    if result["status"] != "FAIL":
        try:
            frozen()
            evidence_paths = [
                p
                for p in (directory / "actions").glob("rosevidence_*.json")
                if not p.name.endswith((".verification.json", ".failed.json"))
            ]
            audits = list(directory.glob("plan-events-*.jsonl"))
            if len(evidence_paths) != 1 or len(audits) != 1:
                raise ValueError("exclusive canonical temporal artifact/observer audit required")
            replay = replay_component_occupancy(
                audits[0],
                json.loads(evidence_paths[0].read_bytes()),
                json.loads((directory / "physics_binding.json").read_bytes()),
            )
            (directory / "closed-source-replay.json").write_text(
                json.dumps(replay, indent=2) + "\n"
            )
            result.update(
                status="PASS",
                task_kernel_succeeded=True,
                physical_acceptance="SIMULATION",
                closed_source_replay="PASS",
                independent_stop="PASS",
            )
        except Exception as exc:
            result.update(status="FAIL", error=type(exc).__name__ + ": " + str(exc))
    (directory / "dynamic-native-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "protocol", "plugin", "vendor-urdf", "native-profile-home"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_episode(
        args.directory, args.protocol, args.plugin, args.vendor_urdf, args.native_profile_home
    )
    print(json.dumps(result), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
