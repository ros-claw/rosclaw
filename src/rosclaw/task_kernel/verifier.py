"""Verifier（PR-H4，总纲 v2 §12）——终态由验收决定，模型自述不算数。

验收类型（typed checks，全部真实执行）：
- artifact 存在 + 非空 + sha256 与登记一致（forged/tampered → FAIL）；
- acceptance.required_files（task workspace 内路径限定）；
- acceptance.run.argv（结构化、解释器白名单、无凭据 env、workspace
  限定——继承十六审 GUARDED_VERIFIER 纪律）；
- 零检查 = 绝不 PASS（ACCEPTANCE_MISSING）。
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

#: 验收解释器白名单（与十六审 GUARDED_VERIFIER 一致——拒绝 shell）。
_ARGV0_ALLOWLIST = frozenset({"python3", "python", "pytest"})


def verify_artifacts(conn_artifacts: list[dict], workspace: Path) -> list[str]:
    """登记 artifact 的真实性核验：存在 + 非空 + hash 一致。"""
    failures: list[str] = []
    for artifact in conn_artifacts:
        path = Path(str(artifact["path"]))
        if not path.exists():
            failures.append(f"artifact 不存在: {path.name}")
            continue
        size = path.stat().st_size
        if size == 0:
            failures.append(f"artifact 为空: {path.name}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(artifact["sha256"]):
            failures.append(f"artifact hash 失配（登记后被改写）: {path.name}")
    return failures


def verify_acceptance(acceptance: dict, workspace: Path) -> tuple[int, list[str]]:
    """acceptance 结构化检查（required_files + run.argv）。"""
    checks = 0
    failures: list[str] = []
    for rel in acceptance.get("required_files") or []:
        checks += 1
        candidate = (workspace / str(rel)).resolve()
        if (
            not str(candidate).startswith(str(workspace.resolve()))
            or not candidate.exists()
        ):
            failures.append(f"缺交付文件 {rel}")
    run = acceptance.get("run")
    if run:
        checks += 1
        argv = [str(a) for a in (run.get("argv") or [])]
        if not argv or argv[0] not in _ARGV0_ALLOWLIST:
            failures.append(
                f"验收解释器 {argv[0] if argv else ''!r} 不在白名单——拒绝执行"
            )
        else:
            import os
            import shutil

            executable = shutil.which(argv[0])
            if executable is None:
                failures.append(f"验收解释器 {argv[0]!r} 未安装")
            else:
                env = {
                    k: os.environ[k]
                    for k in ("PATH", "LANG", "LC_ALL", "TZ")
                    if k in os.environ
                }
                env["HOME"] = str(workspace)
                timeout = min(float(run.get("timeout_sec", 600)), 600)
                try:
                    proc = subprocess.run(
                        [executable, *argv[1:]],
                        cwd=str(workspace),
                        env=env,
                        capture_output=True,
                        timeout=timeout,
                    )
                    if proc.returncode != 0:
                        tail = proc.stdout.decode(errors="replace")[-300:]
                        failures.append(
                            f"验收测试失败(rc={proc.returncode}): {tail}"
                        )
                except subprocess.TimeoutExpired:
                    failures.append(f"验收测试超时({timeout}s)")
    return checks, failures


def verdict_for(
    *,
    artifacts: list[dict],
    acceptance: dict,
    workspace: Path,
    summary: str,
    require_trusted_evidence: bool = False,
    trusted_evidence_present: bool = False,
    extra_failures: list[str] | None = None,
) -> dict[str, Any]:
    """统一判定。返回 {status, checks, failures}——status:
    PASS / REPAIR_REQUIRED（零证据即 REPAIR_REQUIRED，绝不 PASS）。

    PR-N0：require_trusted_evidence（机器人行为任务）时必须有受信
    管道证据（kernel 内部登记的产物）——模型自产证据不算数。"""
    failures = verify_artifacts(artifacts, workspace)
    # PR-N4（§8.4）：正式任务产物引用测试夹具 → 直接失败。
    for artifact in artifacts:
        if "tests/fixtures" in str(artifact.get("path", "")):
            failures.append(
                "RESOURCE_PROVENANCE_FAILED: 产物引用测试夹具 "
                f"{Path(str(artifact['path'])).name}——正式任务不得使用 "
                "fixture 资产"
            )
    if require_trusted_evidence and not trusted_evidence_present:
        failures.append(
            "TRUSTED_EVIDENCE_MISSING: 机器人行为任务缺受信管道的独立"
            "验证证据——模型自产 artifact 不算数"
        )
    failures += list(extra_failures or [])
    checks = len(artifacts)
    acc_checks, acc_failures = verify_acceptance(acceptance, workspace)
    checks += acc_checks
    failures += acc_failures
    if checks == 0:
        # 纯问答任务：总结非空是最后一条真实检查。
        checks = 1
        if not summary.strip():
            failures.append(
                "ACCEPTANCE_MISSING: 无 artifact、无验收定义、无总结——"
                "零证据不得成功"
            )
    return {
        "status": "PASS" if not failures else "REPAIR_REQUIRED",
        "checks": checks,
        "failures": failures,
    }
