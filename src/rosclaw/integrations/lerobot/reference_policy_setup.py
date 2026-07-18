"""Install + smoke-verify the RH56 reference policy worker plugin (P5-E).

After ``rosclaw setup lerobot`` provisions the isolated runtime, the
``--reference-policy rh56`` flow finishes the worker environment:

1. ``pip install`` the bundled ``lerobot-policy-rosclaw-rh56`` plugin into
   the runtime (source checkout or installed ``rosclaw`` package);
2. spawn the persistent worker with the runtime python and verify
   LOAD_POLICY + one INFER round-trip on the bundled reference policy;
3. run an RH56 preflight (transport binding + calibration gate) when the
   profile/calibration files are available.

Everything is local-path based — no dev-machine caches, no manual copies.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _plugin_project_dir() -> Path | None:
    """Locate the installable plugin project (source checkout or wheel)."""
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "worker_plugins" / "lerobot_policy_rosclaw_rh56",
    ]
    # Installed distribution: the wheel force-includes the bare package next
    # to rosclaw; its project metadata lives in the dist-info RECORD, so for
    # installed dists the plugin is already importable and pip install of the
    # project is unnecessary — signal that with a sentinel.
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _plugin_already_importable(runtime_python: Path) -> bool:
    r = subprocess.run(
        [str(runtime_python), "-c", "import lerobot_policy_rosclaw_rh56"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode == 0


def _dataset_extras_importable(runtime_python: Path) -> bool:
    check = (
        "import datasets, pandas, pyarrow, jsonlines, av;"
        "assert tuple(map(int, datasets.__version__.split('.')[:1])) < (5,);"
        "assert tuple(map(int, av.__version__.split('.')[:1])) < (16,)"
    )
    r = subprocess.run(
        [str(runtime_python), "-c", check],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return r.returncode == 0


def _pip_install(runtime_python: Path, *targets: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(runtime_python), "-m", "pip", "install", *targets],
        capture_output=True,
        text=True,
        timeout=900,
    )


def install_rh56_reference_policy(
    runtime_python: Path | str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install the RH56 plugin into the runtime and smoke-verify the policy."""
    runtime_python = Path(runtime_python)
    result: dict[str, Any] = {"runtime_python": str(runtime_python), "steps": []}

    def step(name: str, ok: bool, detail: str = "") -> None:
        result["steps"].append({"name": name, "ok": ok, "detail": detail})

    if not runtime_python.exists():
        step("runtime_python", False, f"missing: {runtime_python}")
        result["ok"] = False
        return result

    # 0. Dataset export extras (pinned to lerobot 0.6.x [dataset] extra) —
    # real practices export to standard LeRobotDataset out of the box.
    if _dataset_extras_importable(runtime_python):
        step("dataset_extras", True, "already importable")
    elif dry_run:
        step("dataset_extras", True, "dry-run: would install lerobot[dataset] pins")
    else:
        proc = _pip_install(
            runtime_python,
            "datasets>=4.7.0,<5.0.0",
            "pandas>=2.0.0,<3.0.0",
            "pyarrow>=21.0.0,<30.0.0",
            "jsonlines",
            "av>=15.0.0,<16.0.0",
        )
        ok = proc.returncode == 0
        step(
            "dataset_extras",
            ok,
            (proc.stdout or proc.stderr).strip().splitlines()[-1][:200]
            if (proc.stdout or proc.stderr).strip()
            else "",
        )
        if not ok:
            result["ok"] = False
            return result

    # 1. Plugin install (skip when the wheel already provides the package).
    if _plugin_already_importable(runtime_python):
        step("plugin_install", True, "already importable (bundled wheel)")
    else:
        project = _plugin_project_dir()
        if project is None:
            step(
                "plugin_install",
                False,
                "plugin project not found (need source checkout or bundled wheel)",
            )
            result["ok"] = False
            return result
        if dry_run:
            step("plugin_install", True, f"dry-run: would pip install {project}")
        else:
            proc = _pip_install(runtime_python, str(project))
            ok = proc.returncode == 0
            step(
                "plugin_install",
                ok,
                (proc.stdout or proc.stderr).strip().splitlines()[-1][:200]
                if (proc.stdout or proc.stderr).strip()
                else "",
            )
            if not ok:
                result["ok"] = False
                return result

    if dry_run:
        step("policy_smoke", True, "dry-run: skipped")
        result["ok"] = True
        return result

    # 2. Worker smoke: LOAD_POLICY + INFER through the persistent runtime.
    try:
        from rosclaw.body.rh56.resources import rh56_reference_policy_path
        from rosclaw.integrations.lerobot.policy_runtime.manager import (
            PersistentRuntimeManager,
        )

        policy_path = str(rh56_reference_policy_path())
        manager = PersistentRuntimeManager(
            python_executable=str(runtime_python),
            policy_path=policy_path,
            device="cpu",
            startup_timeout_sec=120.0,
            timeout_sec=120.0,
        )
        state = manager.start()
        if state.state != "ready":
            step("policy_smoke", False, f"worker not ready: {state.error}")
            result["ok"] = False
            return result
        load = manager.call(
            "LOAD_POLICY",
            {"policy_path": policy_path, "device": "cpu", "allow_network": False},
            timeout_sec=120.0,
        )
        if load.get("status") != "ok":
            step("policy_smoke", False, f"LOAD_POLICY failed: {load}")
            result["ok"] = False
            return result
        manager.call("CREATE_SESSION", {"session_id": "setup_smoke"})
        infer = manager.call(
            "INFER",
            {
                "session_id": "setup_smoke",
                "observation": {
                    "observation.state": [1000.0] * 6,
                    "state_names": ["little", "ring", "middle", "index", "thumb", "thumb_rot"],
                    "task": "hold_current",
                },
                "step_index": 0,
            },
            timeout_sec=30.0,
        )
        manager.stop()
        values = (infer.get("processed_action") or {}).get("values")
        ok = infer.get("status") == "ok" and values and len(values[0]) == 6
        step("policy_smoke", bool(ok), f"INFER values={values}")
        result["ok"] = bool(ok)
    except Exception as exc:  # noqa: BLE001
        step("policy_smoke", False, f"{type(exc).__name__}: {exc}")
        result["ok"] = False
    return result


__all__ = ["install_rh56_reference_policy"]
