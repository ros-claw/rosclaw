"""Verifier 插件注册表（PR-N8，调整方案 §七）。

verdict 由插件链产生——每个插件是一类真实检查；插件可注册/替换，
输出形状不变（PASS / REPAIR_REQUIRED + failures）。终态由验收决定，
模型自述不算数。

既有检查全部落为插件（不再是 verifier.py 里的散装分支）：
- FileArtifactVerifier：artifact 存在 + 非空 + sha256 一致；
- AcceptanceRunVerifier：acceptance.required_files + run.argv
  （解释器白名单/workspace 限定——继承 GUARDED_VERIFIER 纪律）；
- TrustedEvidenceVerifier：行为任务受信证据（N0 熔断）；
- FixtureProhibitionVerifier：产物引用 tests/fixtures（N4）；
- ResourceProvenanceVerifier：执行资源证明闭环（N4.1 四方比对）；
- TrajectoryVerifier：轨迹深度验证（拓扑/闭合/动力学/跟踪/同
  trace ID——五角星不能只验"有 GIF"）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

from rosclaw.task_kernel.verifier import verify_acceptance, verify_artifacts


class VerifierPlugin(Protocol):
    name: str

    def check(self, ctx: dict[str, Any]) -> list[str]:
        """返回失败项列表（空 = 通过）。"""
        ...


class FileArtifactVerifier:
    name = "file_artifact"

    def check(self, ctx: dict[str, Any]) -> list[str]:
        return verify_artifacts(ctx["artifacts"], ctx["workspace"])


class AcceptanceRunVerifier:
    name = "acceptance_run"

    def check(self, ctx: dict[str, Any]) -> list[str]:
        _checks, failures = verify_acceptance(ctx["acceptance"], ctx["workspace"])
        return failures


class TrustedEvidenceVerifier:
    name = "trusted_evidence"

    def check(self, ctx: dict[str, Any]) -> list[str]:
        if ctx.get("require_trusted_evidence") and not ctx.get(
            "trusted_evidence_present"
        ):
            return [
                "TRUSTED_EVIDENCE_MISSING: 机器人行为任务缺受信管道的独立"
                "验证证据——模型自产 artifact 不算数"
            ]
        return []


class FixtureProhibitionVerifier:
    name = "fixture_prohibition"

    def check(self, ctx: dict[str, Any]) -> list[str]:
        failures = []
        for artifact in ctx["artifacts"]:
            if "tests/fixtures" in str(artifact.get("path", "")):
                failures.append(
                    "RESOURCE_PROVENANCE_FAILED: 产物引用测试夹具 "
                    f"{Path(str(artifact['path'])).name}——正式任务不得使用 "
                    "fixture 资产"
                )
        return failures


class ResourceProvenanceVerifier:
    name = "resource_provenance"

    def check(self, ctx: dict[str, Any]) -> list[str]:
        # N4.1 的四方比对在 finish_task 组装（extra_failures 传入）——
        # 插件链直接透传（同一事实源，不重复计算）。
        return list(ctx.get("extra_failures") or [])


class TrajectoryVerifier:
    """轨迹深度验证（五角星事故的直接对策）。

    只验"有 GIF/帧数/误差"不够——必须验：拓扑是声明的形状、轨迹
    闭合、证据是动力学 rollout（SIM_DYN_ROLLOUT）、实际轨迹与规划
    一致、GIF/trace/metrics 同一 trace ID（同目录）。
    """

    name = "trajectory"

    def check(
        self,
        *,
        trace_json: str,
        metrics_json: str,
        gif_path: str,
        home: Path,
        declared_shape: str,
        max_tracking_error_m: float,
    ) -> list[str]:
        failures: list[str] = []
        trace_path = Path(trace_json)
        metrics_path = Path(metrics_json)
        gif = Path(gif_path)
        for p in (trace_path, metrics_path, gif):
            if not p.exists():
                failures.append(f"轨迹证据缺失: {p.name}")
        if failures:
            return failures
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        # 同一 trace ID：三者在同一 traces/<id>/ 目录。
        trace_dir = trace_path.parent.name
        if metrics_path.parent.name != trace_dir or gif.parent.name != trace_dir:
            failures.append(
                f"TRACE_LINKAGE_BROKEN: GIF/trace/metrics 不在同一 trace "
                f"目录（{trace_dir}）——拼凑证据"
            )
        # 动力学确实执行。
        if trace.get("evidence_level") != "SIM_DYN_ROLLOUT":
            failures.append(
                f"NOT_DYNAMICS: 证据等级 {trace.get('evidence_level')!r} "
                "不是 SIM_DYN_ROLLOUT——动力学未执行"
            )
        # 实际轨迹闭合：闭环 = 接触段后段回到"规划路径起点"邻域
        # （actual 含 home→路径起点的转场段与 WP-5 的 lift 抬升段——
        # 闭合语义对接触段测量：接触段后窗内距规划起点的最近距离，
        # 不是 trace 末尾的抬升点，也不是 home）。
        actual = trace.get("actual") or []
        planned = trace.get("planned") or []
        if actual and planned:
            def _path_dist(pt: dict) -> float:
                return min(
                    math.dist(
                        (pt["x"], pt["y"], pt["z"]),
                        (p["x"], p["y"], p["z"]),
                    )
                    for p in planned
                )

            near_idx = [
                i for i, a in enumerate(actual) if _path_dist(a) < 0.05
            ]
            contact = (
                actual[near_idx[0] : near_idx[-1] + 1] if near_idx else actual
            )
            loop_start = planned[0]
            # 接触段后一半（去程必经过起点邻域，回程接近才算闭合）。
            return_leg = contact[len(contact) // 2 :] or contact
            gap = min(
                math.dist(
                    (a["x"], a["y"], a["z"]),
                    (loop_start["x"], loop_start["y"], loop_start["z"]),
                )
                for a in return_leg
            )
            if gap > 0.01:
                failures.append(
                    f"LOOP_NOT_CLOSED: 接触段回程距规划起点最近 "
                    f"{gap:.4f}m——未闭合"
                )
        # 跟踪一致（实际 vs 规划）。
        tracking = metrics.get("tracking") or metrics
        max_err = tracking.get("max_error_m")
        if max_err is None:
            failures.append("TRACKING_MISSING: metrics 缺 max_error_m")
        elif float(max_err) > max_tracking_error_m:
            failures.append(
                f"TRACKING_EXCEEDED: 最大误差 {max_err}m > 阈值 "
                f"{max_tracking_error_m}m"
            )
        # 拓扑是声明的形状（经 plan_hash 反查 plan）。
        plans_dir = Path(home) / "sim" / "plans"
        plan = None
        plan_hash = str(trace.get("plan_hash", ""))
        if plans_dir.exists():
            for plan_file in plans_dir.glob("*.json"):
                candidate = json.loads(plan_file.read_text(encoding="utf-8"))
                if candidate.get("hash") == plan_hash:
                    plan = candidate
                    break
        if plan is None:
            failures.append("PLAN_NOT_FOUND: 按 plan_hash 反查不到 plan")
        else:
            if plan.get("shape") != declared_shape:
                failures.append(
                    f"TOPOLOGY_MISMATCH: plan 形状 {plan.get('shape')!r} != "
                    f"声明 {declared_shape!r}"
                )
            elif declared_shape == "star5":
                failures += self._star_topology_failures(plan)
        return failures

    @staticmethod
    def _star_topology_failures(plan: dict) -> list[str]:
        """五角星拓扑：10 个交替内外顶点 + 内外半径比 ≈ 0.382 +
        闭合。"""
        failures: list[str] = []
        points = plan.get("points") or []
        center = plan.get("center_m") or [0, 0, 0]
        scale = float(plan.get("scale_m") or 0)
        if len(points) < 10 or not scale:
            failures.append("TOPOLOGY_MISMATCH: 五角星顶点不足/无尺度")
            return failures
        radii = [
            math.dist((p["x"], p["y"]), (center[0], center[1]))
            for p in points[:10]
        ]
        outer = max(radii)
        inner = min(radii)
        # 外顶点≈scale，内/外比≈0.381966（黄金分割）。
        if abs(outer - scale) > scale * 0.05:
            failures.append(
                f"TOPOLOGY_MISMATCH: 外半径 {outer:.4f} != 声明尺度 {scale}"
            )
        ratio = inner / outer if outer else 0
        if abs(ratio - 0.381966) > 0.03:
            failures.append(
                f"TOPOLOGY_MISMATCH: 内外半径比 {ratio:.3f} != 0.382——"
                "不是五角星"
            )
        return failures


class VerifierRegistry:
    """插件链——verdict 由链产生；输入上下文一次组装。"""

    def __init__(self, plugins: list[VerifierPlugin]) -> None:
        self.plugins = plugins

    def verdict(self, *, artifacts: list[dict], acceptance: dict,
                workspace: Path, summary: str,
                require_trusted_evidence: bool = False,
                trusted_evidence_present: bool = False,
                extra_failures: list[str] | None = None) -> dict[str, Any]:
        ctx = {
            "artifacts": artifacts,
            "acceptance": acceptance,
            "workspace": workspace,
            "summary": summary,
            "require_trusted_evidence": require_trusted_evidence,
            "trusted_evidence_present": trusted_evidence_present,
            "extra_failures": extra_failures,
        }
        failures: list[str] = []
        for plugin in self.plugins:
            failures += plugin.check(ctx)
        checks = len(artifacts)
        checks += len(acceptance.get("required_files") or [])
        if acceptance.get("run"):
            checks += 1
        if checks == 0:
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


def default_registry() -> VerifierRegistry:
    return VerifierRegistry([
        FileArtifactVerifier(),
        AcceptanceRunVerifier(),
        TrustedEvidenceVerifier(),
        FixtureProhibitionVerifier(),
        ResourceProvenanceVerifier(),
    ])


__all__ = [
    "AcceptanceRunVerifier",
    "FileArtifactVerifier",
    "FixtureProhibitionVerifier",
    "ResourceProvenanceVerifier",
    "TrajectoryVerifier",
    "TrustedEvidenceVerifier",
    "VerifierPlugin",
    "VerifierRegistry",
    "default_registry",
]
