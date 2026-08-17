"""Runtime/Dependency Manager（十六审 P0-C）。

根因：仿真渲染的 Pillow 依赖曾让"开个 Worker 去装 Pillow"——通用 Agent
猜 python、改用户 conda、碰网络运气。环境依赖是 ROSClaw 自己的责任：

- 托管运行时只在 `~/.rosclaw/runtimes/<name>/<digest>/`（venv），
  绝不修改用户 conda/系统 Python；
- 每个能力声明所需包 + readiness probe；digest 覆盖包清单+解释器
  版本+平台——换依赖即换目录（无半成品复用）；
- ensure() 幂等：READY.json 标记 + probe 通过才算就绪；probe 失败
  诚实 RuntimeNotReadyError（调用方映射 BLOCKED，不假成功）；
- pip 安装是 ROSClaw 自有环境的确定性维护（SIM developer policy
  允许），不是 Worker 的自由网络动作。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RuntimeNotReadyError(RuntimeError):
    """托管 runtime 未就绪（诚实失败——调用方映射 BLOCKED）。"""


#: 能力 runtime 声明（包 + readiness probe）。Pillow→PIL 这类
#: 包名≠模块名的映射集中在 _MODULE_NAME_OVERRIDES。
RUNTIME_SPECS: dict[str, dict[str, Any]] = {
    "rosclaw-simulation": {
        "python_packages": ["Pillow>=10"],
        "probe_module": "PIL",
    },
}

_MODULE_NAME_OVERRIDES = {"pillow": "PIL"}


def _module_name_for(package: str) -> str:
    """pip 包名 → import 模块名（best-effort；版本约束剥离）。"""
    import re

    base = re.split(r"[<>=!~\[ ]", package, maxsplit=1)[0].strip()
    lowered = base.lower().replace("-", "_")
    return _MODULE_NAME_OVERRIDES.get(lowered, lowered)


@dataclass(frozen=True)
class RuntimeHandle:
    name: str
    directory: Path
    python: Path
    site_packages: Path
    digest: str

    @property
    def bin_dir(self) -> Path:
        return self.python.parent


class RuntimeManager:
    """`~/.rosclaw/runtimes/` 托管运行时（venv + 声明式包 + probe）。"""

    def __init__(self, home: Path | str) -> None:
        self._root = Path(home) / "runtimes"

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def _digest(spec: dict) -> str:
        payload = json.dumps(
            {
                "packages": sorted(spec.get("python_packages") or []),
                "probe": str(spec.get("probe_module") or ""),
                "py": f"{sys.version_info.major}.{sys.version_info.minor}",
                "platform": platform.machine(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _site_packages(self, directory: Path) -> Path:
        candidates = sorted(directory.glob("lib/python*/site-packages"))
        if not candidates:
            raise RuntimeNotReadyError(f"venv 缺 site-packages: {directory}")
        return candidates[0]

    def _handle(self, name: str, spec: dict) -> RuntimeHandle:
        digest = self._digest(spec)
        directory = self._root / name / digest
        python = directory / "bin" / "python3"
        if not python.exists():
            python = directory / "bin" / "python"
        return RuntimeHandle(
            name=name,
            directory=directory,
            python=python,
            site_packages=(
                self._site_packages(directory) if directory.exists()
                else directory / "lib" / "site-packages"
            ),
            digest=digest,
        )

    def _probe(self, handle: RuntimeHandle, probe_module: str) -> None:
        proc = subprocess.run(
            [str(handle.python), "-c", f"import {probe_module}"],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-300:]
            raise RuntimeNotReadyError(
                f"probe failed: import {probe_module} — {tail}"
            )

    def ensure(self, name: str, spec: dict | None = None) -> RuntimeHandle:
        """确保托管 runtime 就绪（幂等）。失败 → RuntimeNotReadyError。"""
        if spec is None:
            if name not in RUNTIME_SPECS:
                raise RuntimeNotReadyError(f"未知 runtime {name!r}（无声明）")
            spec = RUNTIME_SPECS[name]
        packages = [str(p) for p in (spec.get("python_packages") or [])]
        probe_module = str(
            spec.get("probe_module")
            or (_module_name_for(packages[0]) if packages else "")
        )
        handle = self._handle(name, {**spec, "probe_module": probe_module})
        marker = handle.directory / "READY.json"
        if marker.exists() and handle.python.exists():
            self._probe(handle, probe_module or "sys")
            return handle
        # 重建：清掉半成品目录再建 venv（digest 目录即不可变单元）。
        import shutil

        if handle.directory.exists():
            shutil.rmtree(handle.directory)
        handle.directory.mkdir(parents=True)
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(handle.directory)],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeNotReadyError(
                "venv 创建失败（python3-venv 未安装？）："
                + proc.stderr.decode(errors="replace")[-300:]
            )
        handle = self._handle(name, {**spec, "probe_module": probe_module})
        if packages:
            proc = subprocess.run(
                [str(handle.python), "-m", "pip", "install", *packages],
                capture_output=True,
                timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeNotReadyError(
                    f"托管 runtime 装包失败（网络/索引不可达？）："
                    f"{proc.stderr.decode(errors='replace')[-300:]}"
                )
        if probe_module:
            self._probe(handle, probe_module)
        marker.write_text(
            json.dumps(
                {"name": name, "packages": packages, "digest": handle.digest},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return handle

    def activate(self, handle: RuntimeHandle) -> Path:
        """把托管 site-packages 并入当前进程（同解释器版本由 digest
        保证；幂等）。用于 agentd 进程内的确定性渲染等场景。"""
        path = str(handle.site_packages)
        if path not in sys.path:
            sys.path.insert(0, path)
        return handle.site_packages


def doctor_runtime(home: Path | str, topic: str) -> dict[str, Any]:
    """`doctor <topic>` 的 runtime 报告（无需模型凭据）。"""
    name = f"rosclaw-{topic}"
    spec = RUNTIME_SPECS.get(name)
    if spec is None:
        return {"runtime": name, "ready": False, "error": "未知 runtime 主题"}
    manager = RuntimeManager(home)
    handle = manager._handle(
        name, {**spec, "probe_module": str(spec.get("probe_module") or "")}
    )
    ready_marker = (handle.directory / "READY.json").exists()
    ready = False
    error = ""
    if ready_marker and handle.python.exists():
        try:
            manager._probe(handle, str(spec.get("probe_module") or "sys"))
            ready = True
        except (RuntimeNotReadyError, subprocess.SubprocessError) as exc:
            error = str(exc)[:300]
    return {
        "runtime": name,
        "directory": str(handle.directory),
        "packages": list(spec.get("python_packages") or []),
        "ready": ready,
        **({"error": error} if error else {}),
        "hint": "" if ready else "首次使用仿真能力时会自动预置（托管 venv）",
    }
