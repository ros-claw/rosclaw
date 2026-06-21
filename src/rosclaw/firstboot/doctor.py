"""Structured health diagnosis for ROSClaw First Boot."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from rosclaw import __version__

from .config import FirstbootConfig, generate_rosclaw_yaml, load_rosclaw_yaml
from .mcp import generate_mcp_config
from .telemetry import generate_telemetry_yaml
from .workspace import ensure_minimal_workspace


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class DoctorStatus(StrEnum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass
class CheckResult:
    id: str
    name: str
    status: CheckStatus
    required: bool
    message: str
    fix: str | None = None


@dataclass
class DoctorResult:
    status: DoctorStatus
    exit_code: int
    checks: list[CheckResult]


CORE_MODULES = [
    ("rosclaw.core.runtime", "Runtime"),
    ("rosclaw.core.event_bus", "EventBus"),
    ("rosclaw.provider.core.registry", "ProviderRegistry"),
    ("rosclaw.sandbox.runtime_adapter", "SandboxRuntimeAdapter"),
    ("rosclaw.memory.interface", "MemoryInterface"),
    ("rosclaw.practice.episode_recorder", "EpisodeRecorder"),
    ("rosclaw.how.engine", "HeuristicEngine"),
]


class FirstbootDoctor:
    """Doctor implementation supporting bootstrap, full, fix and JSON output modes."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path.home() / ".rosclaw"

    def run_bootstrap(
        self,
        *,
        fix: bool = False,
        json_output: bool = False,
    ) -> DoctorResult:
        """L0 bootstrap checks."""
        checks = self._collect_bootstrap_checks()
        if fix:
            self._auto_fix(checks)
        return self._compile(checks, json_output=json_output)

    def run_full(
        self,
        *,
        fix: bool = False,
        json_output: bool = False,
        check_gpu: bool = False,
        check_network: bool = False,
    ) -> DoctorResult:
        """L1 core + L2 optional checks."""
        checks = self._collect_bootstrap_checks()
        checks.extend(self._check_core_modules())
        checks.extend(self._check_config_schema())
        checks.extend(self._check_directories())
        checks.extend(self._check_eurdf_zoo())
        checks.extend(self._check_provider_registry())
        checks.extend(self._check_sandbox())
        checks.extend(self._check_practice())
        checks.extend(self._check_memory())
        checks.extend(self._check_docker())
        checks.extend(self._check_ros2())
        checks.extend(self._check_mujoco())
        if check_gpu:
            checks.extend(self._check_cuda())
        checks.extend(self._check_git())
        if check_network:
            checks.extend(self._check_network())
        checks.extend(self._check_mcp_server())

        if fix:
            self._auto_fix(checks)
        return self._compile(checks, json_output=json_output)

    def _collect_bootstrap_checks(self) -> list[CheckResult]:
        """Collect L0 bootstrap checks without printing."""
        return [
            self._check_cli(),
            self._check_python(),
            self._check_workspace_writable(),
            self._check_permissions(),
            self._check_install_json(),
            self._check_config_dir(),
            self._check_mcp_config(),
            self._check_telemetry_config(),
        ]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_cli(self) -> CheckResult:
        rosclaw_path = shutil.which("rosclaw")
        if rosclaw_path:
            return CheckResult(
                "core.cli",
                "ROSClaw CLI",
                CheckStatus.PASS,
                True,
                f"rosclaw {__version__} at {rosclaw_path}",
            )
        return CheckResult(
            "core.cli",
            "ROSClaw CLI",
            CheckStatus.FAIL,
            True,
            "rosclaw not found in PATH",
            "Re-run the bootstrap script or add ~/.rosclaw/bin to PATH",
        )

    def _check_python(self) -> CheckResult:
        version = platform.python_version()
        ok = (platform.python_version_tuple()[0], platform.python_version_tuple()[1]) >= ("3", "11")
        if ok:
            return CheckResult("core.python", "Python version", CheckStatus.PASS, True, version)
        return CheckResult(
            "core.python",
            "Python version",
            CheckStatus.FAIL,
            True,
            f"{version} < 3.11",
            "Install Python 3.11+ (Ubuntu: apt install python3.11 python3.11-venv)",
        )

    def _check_workspace_writable(self) -> CheckResult:
        try:
            self.home.mkdir(parents=True, exist_ok=True)
            ok = os.access(self.home, os.W_OK)
        except OSError:
            ok = False
        if ok:
            return CheckResult(
                "core.workspace",
                "Workspace writable",
                CheckStatus.PASS,
                True,
                str(self.home),
            )
        return CheckResult(
            "core.workspace",
            "Workspace writable",
            CheckStatus.FAIL,
            True,
            str(self.home),
            "Create the directory or set ROSCLAW_HOME to a writable path",
        )

    def _check_install_json(self) -> CheckResult:
        path = self.home / "state" / "install.json"
        if path.exists():
            return CheckResult(
                "core.install_json",
                "Install metadata",
                CheckStatus.PASS,
                False,
                "found",
            )
        return CheckResult(
            "core.install_json",
            "Install metadata",
            CheckStatus.WARN,
            False,
            "missing",
            "Run the bootstrap script",
        )

    def _check_config_dir(self) -> CheckResult:
        path = self.home / "config"
        if path.exists():
            return CheckResult(
                "core.config_dir",
                "Config directory",
                CheckStatus.PASS,
                True,
                str(path),
            )
        return CheckResult(
            "core.config_dir",
            "Config directory",
            CheckStatus.FAIL,
            True,
            str(path),
            "Run `rosclaw firstboot`",
        )

    def _check_permissions(self) -> CheckResult:
        try:
            mode = self.home.stat().st_mode
            if mode & 0o077:
                return CheckResult(
                    "core.permissions",
                    "Workspace permissions",
                    CheckStatus.WARN,
                    False,
                    oct(mode & 0o777),
                    "Run `rosclaw doctor --fix`",
                )
        except OSError as exc:
            return CheckResult(
                "core.permissions",
                "Workspace permissions",
                CheckStatus.WARN,
                False,
                str(exc),
                "Run `rosclaw doctor --fix`",
            )
        return CheckResult(
            "core.permissions",
            "Workspace permissions",
            CheckStatus.PASS,
            False,
            "OK",
        )

    def _check_mcp_config(self) -> CheckResult:
        cfg = load_rosclaw_yaml(self.home)
        mcp_enabled = cfg.get("mcp", {}).get("enabled", True)
        path = self.home / "config" / "mcp.json"
        if path.exists():
            return CheckResult(
                "core.mcp_config",
                "MCP config",
                CheckStatus.PASS,
                False,
                str(path),
            )
        if not mcp_enabled:
            return CheckResult(
                "core.mcp_config",
                "MCP config",
                CheckStatus.SKIP,
                False,
                "MCP disabled",
            )
        return CheckResult(
            "core.mcp_config",
            "MCP config",
            CheckStatus.WARN,
            False,
            "missing",
            "Run `rosclaw doctor --fix`",
        )

    def _check_telemetry_config(self) -> CheckResult:
        path = self.home / "config" / "telemetry.yaml"
        if path.exists():
            return CheckResult(
                "core.telemetry_config",
                "Telemetry config",
                CheckStatus.PASS,
                False,
                str(path),
            )
        return CheckResult(
            "core.telemetry_config",
            "Telemetry config",
            CheckStatus.WARN,
            False,
            "missing",
            "Run `rosclaw doctor --fix`",
        )

    def _check_core_modules(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for mod_name, cls_name in CORE_MODULES:
            check_id = f"core.module.{mod_name.replace('.', '_')}"
            try:
                mod = importlib.import_module(mod_name)
                getattr(mod, cls_name)
                results.append(
                    CheckResult(
                        check_id,
                        f"Module {mod_name}",
                        CheckStatus.PASS,
                        True,
                        "OK",
                    )
                )
            except Exception as exc:
                results.append(
                    CheckResult(
                        check_id,
                        f"Module {mod_name}",
                        CheckStatus.FAIL,
                        True,
                        str(exc),
                        "pip install -e .",
                    )
                )
        return results

    def _check_config_schema(self) -> list[CheckResult]:
        from .config import validate_config

        valid, errors = validate_config(self.home)
        if valid:
            return [
                CheckResult(
                    "core.config_schema",
                    "Config schema",
                    CheckStatus.PASS,
                    True,
                    "rosclaw.yaml valid",
                )
            ]
        return [
            CheckResult(
                "core.config_schema",
                "Config schema",
                CheckStatus.FAIL,
                True,
                "; ".join(errors),
                "Run `rosclaw firstboot` or `rosclaw doctor --fix`",
            )
        ]

    def _check_directories(self) -> list[CheckResult]:
        from .workspace import DEFAULT_DIRS

        results: list[CheckResult] = []
        for rel in DEFAULT_DIRS:
            path = self.home / rel
            if path.exists():
                results.append(
                    CheckResult(
                        f"core.dir.{rel.replace('/', '_')}",
                        f"Directory {rel}",
                        CheckStatus.PASS,
                        False,
                        "OK",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        f"core.dir.{rel.replace('/', '_')}",
                        f"Directory {rel}",
                        CheckStatus.WARN,
                        False,
                        "missing",
                        f"Run `rosclaw doctor --fix` to create {rel}",
                    )
                )
        return results

    def _check_eurdf_zoo(self) -> list[CheckResult]:
        candidates = [
            self.home / "robots" / "e-urdf-zoo",
            Path(__file__).parent.parent.parent.parent / "e-urdf-zoo",
        ]
        for candidate in candidates:
            if candidate.exists() and any(candidate.iterdir()):
                return [
                    CheckResult(
                        "core.eurdf_zoo",
                        "e-URDF-Zoo",
                        CheckStatus.PASS,
                        False,
                        str(candidate),
                    )
                ]
        return [
            CheckResult(
                "core.eurdf_zoo",
                "e-URDF-Zoo",
                CheckStatus.WARN,
                False,
                "not found",
                "Install robot profiles or set robot_zoo_path in rosclaw.yaml",
            )
        ]

    def _check_provider_registry(self) -> list[CheckResult]:
        try:
            from rosclaw.provider.core.registry import ProviderRegistry

            registry = ProviderRegistry()
            providers = list(registry.list_providers()) if hasattr(registry, "list_providers") else []
            return [
                CheckResult(
                    "runtime.provider_registry",
                    "Provider registry",
                    CheckStatus.PASS,
                    True,
                    f"{len(providers)} provider(s)",
                )
            ]
        except Exception as exc:
            return [
                CheckResult(
                    "runtime.provider_registry",
                    "Provider registry",
                    CheckStatus.FAIL,
                    True,
                    str(exc),
                    "pip install -e .",
                )
            ]

    def _check_sandbox(self) -> list[CheckResult]:
        try:
            from rosclaw.sandbox.runtime_adapter import SandboxRuntimeAdapter

            return [
                CheckResult(
                    "runtime.sandbox",
                    "Sandbox module",
                    CheckStatus.PASS,
                    True,
                    SandboxRuntimeAdapter.__name__,
                )
            ]
        except Exception as exc:
            return [
                CheckResult(
                    "runtime.sandbox",
                    "Sandbox module",
                    CheckStatus.FAIL,
                    True,
                    str(exc),
                    "pip install -e .",
                )
            ]

    def _check_practice(self) -> list[CheckResult]:
        try:
            from rosclaw.practice.episode_recorder import EpisodeRecorder

            recorder = EpisodeRecorder("doctor", event_bus=None, artifact_base_dir=str(self.home / "artifacts"))
            episodes = recorder.list_episodes()
            return [
                CheckResult(
                    "runtime.practice",
                    "Practice recorder",
                    CheckStatus.PASS,
                    True,
                    f"OK ({len(episodes)} episodes)",
                )
            ]
        except Exception as exc:
            return [
                CheckResult(
                    "runtime.practice",
                    "Practice recorder",
                    CheckStatus.FAIL,
                    True,
                    str(exc),
                    "pip install -e .",
                )
            ]

    def _check_memory(self) -> list[CheckResult]:
        try:
            from rosclaw.memory.interface import MemoryInterface

            mem = MemoryInterface("doctor")
            mem._do_initialize()
            stats = mem.get_statistics()
            return [
                CheckResult(
                    "runtime.memory",
                    "Memory backend",
                    CheckStatus.PASS,
                    True,
                    f"OK ({stats.get('total_experiences', 0)} experiences)",
                )
            ]
        except Exception as exc:
            return [
                CheckResult(
                    "runtime.memory",
                    "Memory backend",
                    CheckStatus.FAIL,
                    True,
                    str(exc),
                    "pip install -e .",
                )
            ]

    def _check_docker(self) -> list[CheckResult]:
        docker_path = shutil.which("docker")
        if docker_path:
            return [
                CheckResult(
                    "optional.docker",
                    "Docker",
                    CheckStatus.PASS,
                    False,
                    docker_path,
                )
            ]
        return [
            CheckResult(
                "optional.docker",
                "Docker",
                CheckStatus.WARN,
                False,
                "not found",
                "Container demos unavailable. Install Docker to enable them.",
            )
        ]

    def _check_ros2(self) -> list[CheckResult]:
        ros2_path = shutil.which("ros2")
        if not ros2_path:
            return [
                CheckResult(
                    "optional.ros2",
                    "ROS 2",
                    CheckStatus.WARN,
                    False,
                    "not found",
                    "source /opt/ros/humble/setup.bash or install ROS 2",
                )
            ]
        distro = os.environ.get("ROS_DISTRO", "")
        if distro:
            return [
                CheckResult(
                    "optional.ros2",
                    "ROS 2",
                    CheckStatus.PASS,
                    False,
                    f"{distro} @ {ros2_path}",
                )
            ]
        return [
            CheckResult(
                "optional.ros2",
                "ROS 2",
                CheckStatus.WARN,
                False,
                "ros2 found but ROS_DISTRO not set",
                "source /opt/ros/<distro>/setup.bash",
            )
        ]

    def _check_mujoco(self) -> list[CheckResult]:
        try:
            import mujoco

            return [
                CheckResult(
                    "optional.mujoco",
                    "MuJoCo",
                    CheckStatus.PASS,
                    False,
                    mujoco.__version__,
                )
            ]
        except ImportError:
            return [
                CheckResult(
                    "optional.mujoco",
                    "MuJoCo",
                    CheckStatus.WARN,
                    False,
                    "not installed",
                    "pip install mujoco",
                )
            ]

    def _check_cuda(self) -> list[CheckResult]:
        try:
            import torch

            if torch.cuda.is_available():
                return [
                    CheckResult(
                        "optional.cuda",
                        "CUDA",
                        CheckStatus.PASS,
                        False,
                        f"{torch.cuda.device_count()} device(s)",
                    )
                ]
            return [
                CheckResult(
                    "optional.cuda",
                    "CUDA",
                    CheckStatus.WARN,
                    False,
                    "PyTorch installed but CUDA not available",
                    "Install NVIDIA drivers and CUDA toolkit",
                )
            ]
        except ImportError:
            return [
                CheckResult(
                    "optional.cuda",
                    "CUDA",
                    CheckStatus.WARN,
                    False,
                    "PyTorch not installed",
                    "pip install torch",
                )
            ]

    def _check_git(self) -> list[CheckResult]:
        git_path = shutil.which("git")
        if git_path:
            return [
                CheckResult(
                    "optional.git",
                    "Git",
                    CheckStatus.PASS,
                    False,
                    git_path,
                )
            ]
        return [
            CheckResult(
                "optional.git",
                "Git",
                CheckStatus.WARN,
                False,
                "not found",
                "Install git to enable SDK-to-MCP workflows",
            )
        ]

    def _check_network(self) -> list[CheckResult]:
        try:
            result = subprocess.run(
                ["curl", "-sI", "https://pypi.org"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            ok = result.returncode == 0
        except Exception:
            ok = False

        if ok:
            return [
                CheckResult(
                    "optional.network",
                    "Network",
                    CheckStatus.PASS,
                    False,
                    "PyPI reachable",
                )
            ]
        return [
            CheckResult(
                "optional.network",
                "Network",
                CheckStatus.WARN,
                False,
                "PyPI not reachable",
                "Offline mode available; cloud features require network",
            )
        ]

    def _check_mcp_server(self) -> list[CheckResult]:
        rosclaw_mcp_path = shutil.which("rosclaw-mcp")
        if rosclaw_mcp_path:
            return [
                CheckResult(
                    "optional.mcp_server",
                    "MCP server",
                    CheckStatus.PASS,
                    False,
                    rosclaw_mcp_path,
                )
            ]
        return [
            CheckResult(
                "optional.mcp_server",
                "MCP server",
                CheckStatus.WARN,
                False,
                "rosclaw-mcp not in PATH",
                "Install the rosclaw package with console scripts",
            )
        ]

    # ------------------------------------------------------------------
    # Fix logic
    # ------------------------------------------------------------------

    def _auto_fix(self, checks: list[CheckResult]) -> None:
        """Apply safe fixes only."""
        for check in checks:
            if check.status != CheckStatus.FAIL and check.status != CheckStatus.WARN:
                continue

            if check.id == "core.config_dir":
                (self.home / "config").mkdir(parents=True, exist_ok=True)
                check.status = CheckStatus.PASS
                check.message = "created"
                check.fix = None
            elif check.id == "core.workspace":
                self.home.mkdir(parents=True, exist_ok=True)
                check.status = CheckStatus.PASS
                check.message = "created"
                check.fix = None
            elif check.id.startswith("core.dir."):
                rel = check.id.replace("core.dir.", "").replace("_", "/")
                (self.home / rel).mkdir(parents=True, exist_ok=True)
                check.status = CheckStatus.PASS
                check.message = "created"
                check.fix = None
            elif check.id == "core.install_json":
                ensure_minimal_workspace(self.home)
                check.status = CheckStatus.PASS
                check.message = "regenerated"
                check.fix = None
            elif check.id == "core.config_schema":
                config = FirstbootConfig(workspace={"home": str(self.home)})
                config.apply_profile("offline")
                generate_rosclaw_yaml(self.home, config)
                check.status = CheckStatus.PASS
                check.message = "regenerated"
                check.fix = None
            elif check.id == "core.mcp_config":
                generate_mcp_config(self.home)
                check.status = CheckStatus.PASS
                check.message = "created"
                check.fix = None
            elif check.id == "core.telemetry_config":
                generate_telemetry_yaml(self.home, enabled=False)
                check.status = CheckStatus.PASS
                check.message = "created"
                check.fix = None
            elif check.id == "core.cli":
                self._create_path_shim(check)
            elif check.id == "core.permissions":
                self._tighten_permissions(check)

    def _create_path_shim(self, check: CheckResult) -> None:
        """Create a PATH shim so the rosclaw command is reachable."""
        python = sys.executable or shutil.which("python3") or shutil.which("python")
        if not python:
            return
        try:
            subprocess.run(
                [python, "-c", "import rosclaw"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return

        shim_dir = self.home / "bin"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim = shim_dir / "rosclaw"
        shim.write_text(
            f'#!/usr/bin/env bash\nexec "{python}" -m rosclaw.cli "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(0o755)
        check.status = CheckStatus.WARN
        check.message = f"rosclaw shim created at {shim}; add {shim_dir} to PATH"
        check.fix = f'export PATH="{shim_dir}:$PATH"'

    def _tighten_permissions(self, check: CheckResult) -> None:
        """Restrict access to sensitive workspace directories."""
        try:
            for path in (
                self.home,
                self.home / "config",
                self.home / "state",
                self.home / "logs",
            ):
                if path.exists():
                    path.chmod(0o700)
            check.status = CheckStatus.PASS
            check.message = "tightened"
            check.fix = None
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Compilation / output
    # ------------------------------------------------------------------

    def _compile(
        self,
        checks: list[CheckResult],
        json_output: bool,
    ) -> DoctorResult:
        required_fails = [c for c in checks if c.status == CheckStatus.FAIL and c.required]
        required_warns = [c for c in checks if c.status == CheckStatus.WARN and c.required]
        optional_issues = [c for c in checks if c.status in (CheckStatus.WARN, CheckStatus.FAIL) and not c.required]

        if required_fails:
            status = DoctorStatus.FAILED
            exit_code = 1
        elif required_warns:
            status = DoctorStatus.DEGRADED
            exit_code = 2
        elif optional_issues:
            status = DoctorStatus.READY_WITH_WARNINGS
            exit_code = 0
        else:
            status = DoctorStatus.READY
            exit_code = 0

        result = DoctorResult(status=status, exit_code=exit_code, checks=checks)

        if json_output:
            self._print_json(result)
        else:
            self._print_human(result)

        return result

    def _print_json(self, result: DoctorResult) -> None:
        data = {
            "status": result.status.value,
            "exit_code": result.exit_code,
            "checks": [asdict(c) for c in result.checks],
        }
        print(json.dumps(data, indent=2, default=str))

    def _print_human(self, result: DoctorResult) -> None:
        print("=" * 60)
        print("ROSClaw Doctor")
        print("=" * 60)

        categories: dict[str, list[CheckResult]] = {"Core": [], "Runtime": [], "Optional": []}
        for c in result.checks:
            if c.id.startswith("core."):
                categories["Core"].append(c)
            elif c.id.startswith("runtime."):
                categories["Runtime"].append(c)
            else:
                categories["Optional"].append(c)

        for category, items in categories.items():
            if not items:
                continue
            print(f"\n{category}:")
            for c in items:
                icon = "✅" if c.status == CheckStatus.PASS else "⚠️" if c.status == CheckStatus.WARN else "❌"
                print(f"  {icon} {c.name:<30} {c.message}")

        print("\n" + "=" * 60)
        print(f"Result: {result.status.value}")
        if result.status != DoctorStatus.READY:
            fixes = [c for c in result.checks if c.status != CheckStatus.PASS and c.fix]
            if fixes:
                print("\nRecommended fixes:")
                for c in fixes:
                    print(f"  • {c.name}: {c.fix}")
        print("=" * 60)


def render_legacy_doctor_report(checks: list[tuple[str, str, bool]], issues: list[str]) -> int:
    """Render the legacy human-readable doctor report for backward compatibility.

    This is used by the unqualified `rosclaw doctor` path.
    """
    print("=" * 60)
    print("ROSClaw v1.0 — Doctor")
    print("=" * 60)
    for name, value, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name:<30} {value}")
    print("=" * 60)

    if issues:
        print(f"\n⚠️  Issues found ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRecommendations:")
        if any("e-URDF-Zoo" in i for i in issues):
            print("  • Ensure e-urdf-zoo/ directory exists alongside src/")
        if any("rosclaw.yaml" in i for i in issues):
            print("  • Run: rosclaw firstboot")
        if any("Not installed" in i for i in issues):
            print("  • Install deps: pip install -e .")
        return 1

    print("\n✅ All checks passed. ROSClaw is healthy!")
    return 0
