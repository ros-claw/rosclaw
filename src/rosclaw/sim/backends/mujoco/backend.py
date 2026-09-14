"""MujocoBackend（PR-MH2/MH3，ADR-0014，规格 §12/§14-§16）。

Maturity: experimental（ADR-0000 §4）。

模型服务：load_model / inspect_model / compile_model / patch_model。
状态与实验（MH3）：initial_state / snapshot_state / restore_state /
capture_and_store / fork_state / rollout / observe。

存储设计（双对象，全部经 MH1 SimStore 不可变落盘）：
- ``models/<simmdl_*>.json`` = model manifest（mjcf_xml + assets ref
  表 + source + parent_model_ref + patches + compile_warnings），
  无 created_at 字段 → 内容寻址天然幂等（同 patch 重放同 ref）；
- ``models/<simmdl_*>.bin`` = mesh/texture 资产原始字节（内容寻址，
  母子模型自动去重）；
- created_at 由 manifest 落盘 mtime 派生（首次创建时间，幂等稳定）。

运行时 MjModel 不进 store——store 只存可序列化负载，模型由
compile_model 现编译。无网络资源（规格 §12.1）。
"""

from __future__ import annotations

import contextlib
import hashlib
import posixpath
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.contracts.common import canonical_json, content_hash
from rosclaw.sim.audit.context import AuditContext
from rosclaw.sim.audit.engine import run_checks
from rosclaw.sim.audit.policy import STRICT_POLICY, AuditPolicy
from rosclaw.sim.backends.mujoco import observe as observe_mod
from rosclaw.sim.backends.mujoco import rollout as rollout_mod
from rosclaw.sim.backends.mujoco import state
from rosclaw.sim.backends.mujoco.inspect import inspect_model_full
from rosclaw.sim.backends.mujoco.patch import apply_patches
from rosclaw.sim.backends.mujoco.rollout import DEFAULT_BUDGETS
from rosclaw.sim.capabilities import probe_mujoco_capabilities
from rosclaw.sim.contracts import (
    AuditResult,
    ComparisonResult,
    ModelInspection,
    ModelPatchResult,
    ModelReference,
    ObservationResult,
    SimulationBackendCapabilities,
    SimulationReceipt,
    SimulationTrace,
)
from rosclaw.sim.experiment import compare as compare_mod
from rosclaw.sim.experiment.metrics import MetricCollector
from rosclaw.sim.model_inspect import inspect_mjcf
from rosclaw.sim.resolve import resolve_mjcf_source, source_kind_for
from rosclaw.sim.store import SimStore


class MujocoBackend:
    """MuJoCo 物理仿真后端（phase-1：模型服务）。"""

    name = "mujoco"

    def __init__(self, task_root: Path | str) -> None:
        self._task_root = Path(task_root)
        self.store = SimStore(task_root)

    # -- 能力 ---------------------------------------------------------------

    def capabilities(self) -> SimulationBackendCapabilities:
        return probe_mujoco_capabilities()

    # -- 模型服务 -----------------------------------------------------------

    def load_model(self, asset_ref: str) -> ModelReference:
        """加载 task-local / e-URDF zoo MJCF：编译冒烟 + 资产捕获 + 落盘。"""
        import mujoco  # noqa: F401

        path = resolve_mjcf_source(asset_ref, task_root=self._task_root)
        xml_bytes = path.read_bytes()
        try:
            body_description = inspect_mjcf(path).to_dict()
        except ValueError as exc:
            raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc
        ref = self.load_model_xml(
            xml_bytes.decode("utf-8"),
            source={
                "kind": source_kind_for(path, task_root=self._task_root),
                "ref": str(asset_ref),
            },
            assets=self._assets_from_file(path),
            body_description=body_description,
        )
        return ref.model_copy(
            update={"runtime_capabilities": self.capabilities().to_canonical_dict()}
        )

    def load_model_xml(
        self,
        xml_text: str,
        *,
        source: dict[str, Any],
        assets: dict[str, bytes] | None = None,
        extra_manifest: dict[str, Any] | None = None,
        body_description: dict[str, Any] | None = None,
    ) -> ModelReference:
        """从 XML 文本装载模型（WorldSpec 编译产物 / patch 之外的生成源）。"""
        import mujoco

        xml_bytes = xml_text.encode("utf-8")
        try:
            spec = mujoco.MjSpec.from_string(xml_text, assets=assets or None)
        except Exception as exc:
            raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc
        warnings = self._compile_with_warnings(spec)

        asset_refs: dict[str, str] = {}
        for key, data in (assets or {}).items():
            asset_refs[key] = self.store.put("models", data)

        manifest = {
            "kind": "model_manifest",
            "mjcf_xml": xml_text,
            "assets": asset_refs,
            "source": source,
            "parent_model_ref": None,
            "patches": [],
            "compile_warnings": warnings,
            "backend": "mujoco",
            "backend_version": str(mujoco.__version__),
            **(extra_manifest or {}),
        }
        ref = self.store.put("models", manifest)
        return ModelReference(
            backend="mujoco",
            backend_version=str(mujoco.__version__),
            created_at=self._created_at(ref),
            model_ref=ref,
            model_digest="sha256:" + hashlib.sha256(xml_bytes).hexdigest(),
            source=source,
            compiled=True,
            body_description=body_description or {},
            compile_warnings=warnings,
        )

    def _assets_from_file(self, path: Path) -> dict[str, bytes]:
        """从磁盘 MJCF 捕获 file 引用资产（containment 校验）。"""
        import mujoco

        try:
            spec = mujoco.MjSpec.from_file(str(path))
        except Exception as exc:
            raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc
        return self._capture_assets(spec, path.parent)

    def inspect_model(self, model_ref: str) -> ModelInspection:
        """编译 + §12.2 全量检查（summary 仅供理解，结构化字段权威）。"""
        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        detail = inspect_model_full(model, model_digest=self._xml_digest(manifest), spec=spec)
        return ModelInspection(
            backend="mujoco",
            backend_version=manifest["backend_version"],
            model_ref=model_ref,
            nq=detail["nq"],
            nv=detail["nv"],
            nu=detail["nu"],
            joints=detail["joints_detail"],
            actuators=detail["actuators_detail"],
            sensors=[s["name"] for s in detail["sensors_detail"]],
            cameras=[c["name"] for c in detail["cameras_detail"]],
            sites=[s["name"] for s in detail["sites_detail"]],
            summary=detail["summary_cn"],
            detail=detail,
        )

    def compile_model(self, model_ref: str) -> ModelInspection:
        """显式编译冒烟 + 全量检查（MH2 与 inspect_model 同语义）。"""
        return self.inspect_model(model_ref)

    def describe_model(self, model_ref: str) -> ModelReference:
        """从不可变 manifest 重建 ModelReference（含亲缘）。"""
        manifest = self._manifest(model_ref)
        return ModelReference(
            backend=manifest["backend"],
            backend_version=manifest["backend_version"],
            created_at=self._created_at(model_ref),
            model_ref=model_ref,
            model_digest=self._xml_digest(manifest),
            parent_model_ref=manifest["parent_model_ref"],
            source=manifest["source"],
            compiled=True,
        )

    def patch_model(self, model_ref: str, patches: list[dict[str, Any]]) -> ModelPatchResult:
        """MjSpec 结构化补丁：apply → compile → 新 manifest（母模型不动）。"""
        import mujoco

        manifest = self._manifest(model_ref)
        spec = mujoco.MjSpec.from_string(
            manifest["mjcf_xml"], assets=self._load_assets(manifest) or None
        )
        apply_patches(spec, patches)  # MODEL_PATCH_INVALID / TARGET_NOT_FOUND / FIELD_UNSUPPORTED
        self._compile_with_warnings(spec)  # 失败 MODEL_COMPILE_FAILED，母模型分毫不动

        child = {
            **manifest,
            "mjcf_xml": spec.to_xml(),
            "parent_model_ref": model_ref,
            "patches": patches,
        }
        new_ref = self.store.put("models", child)
        return ModelPatchResult(
            backend="mujoco",
            backend_version=manifest["backend_version"],
            created_at=datetime.now(UTC).isoformat(),
            patch_digest=content_hash("simpat", patches),
            ok=True,
            new_model_ref=new_ref,
        ).with_digest()

    # -- 状态：快照 / 恢复 / 分叉（MH3，规格 §9/§14） -------------------------

    def snapshot_state(self, model_ref: str, values: dict[str, Any]) -> str:
        """校验并不可变落盘一个状态快照，返回 state_ref。"""
        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        if values.get("model_digest") != self._xml_digest(manifest):
            raise ValueError(
                f"CROSS_MODEL_REF: state digest {values.get('model_digest')!r} != model digest"
            )
        state.validate_state_values(model, values)
        payload = {
            "kind": "state_snapshot",
            "model_ref": model_ref,
            "model_digest": self._xml_digest(manifest),
            **{key: values[key] for key in ("time", *state.STATE_ARRAY_KEYS)},
        }
        return self.store.put("states", payload)

    def initial_state(self, model_ref: str) -> str:
        """模型的初始状态（qpos0 + mj_forward）快照，确定性幂等。"""
        import mujoco

        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return self.capture_and_store(model_ref, model, data)

    def capture_and_store(self, model_ref: str, model, data) -> str:  # noqa: ANN001
        """捕获 MjData 当前状态并落盘。"""
        manifest = self._manifest(model_ref)
        values = state.capture_state(model, data)
        values["model_digest"] = self._xml_digest(manifest)
        payload = {
            "kind": "state_snapshot",
            "model_ref": model_ref,
            "model_digest": values["model_digest"],
            **{key: values[key] for key in ("time", *state.STATE_ARRAY_KEYS)},
        }
        return self.store.put("states", payload)

    def restore_state(self, model_ref: str, state_ref: str):  # noqa: ANN202
        """恢复状态为 (MjModel, MjData)；跨模型 / 维度 / 非有限 fail closed。"""
        import mujoco

        manifest = self._manifest(model_ref)
        snap = self.store.get(state_ref)
        if not isinstance(snap, dict) or snap.get("kind") != "state_snapshot":
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")
        if snap.get("model_digest") != self._xml_digest(manifest):
            raise ValueError(
                f"CROSS_MODEL_REF: state {state_ref!r} does not belong to {model_ref!r}"
            )
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        data = mujoco.MjData(model)
        state.apply_state(model, data, snap)
        return model, data

    def fork_state(self, model_ref: str, state_ref: str, n: int) -> dict[str, Any]:
        """从同一状态分叉 N 个实验分支；branch 初始 digest 必然一致。"""
        manifest = self._manifest(model_ref)
        snap = self.store.get(state_ref)
        if not isinstance(snap, dict) or snap.get("kind") != "state_snapshot":
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")
        if snap.get("model_digest") != self._xml_digest(manifest):
            raise ValueError("CROSS_MODEL_REF: fork base state does not belong to model")
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or n < 1
            or n > DEFAULT_BUDGETS["max_branch_count"]
        ):
            raise ValueError(
                f"SIM_BUDGET_EXCEEDED: branch count {n!r} outside [1, {DEFAULT_BUDGETS['max_branch_count']}]"
            )
        record = {
            "kind": "fork",
            "model_ref": model_ref,
            "base_state_ref": state_ref,
            "n": n,
            "branch_refs": [state_ref] * n,
            "branches": [{"branch_id": f"b{i}", "state_ref": state_ref} for i in range(n)],
        }
        record["fork_ref"] = self.store.put("experiments", record)
        return record

    def transplant_state(self, model_ref: str, state_ref: str) -> str:
        """显式跨模型状态移植（参数实验场景：同物理状态 → patch 后模型）。

        与 restore 的 fail-closed 不同，这是**显式操作**：维度必须完全
        一致（nq/nv/na/nu/nmocap），否则 STATE_DIMENSION；移植记录
        provenance（transplanted_from）。绝不静默截断或补零。
        """
        manifest = self._manifest(model_ref)
        snap = self.store.get(state_ref)
        if not isinstance(snap, dict) or snap.get("kind") != "state_snapshot":
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        state.validate_state_values(model, snap)
        payload = {
            "kind": "state_snapshot",
            "model_ref": model_ref,
            "model_digest": self._xml_digest(manifest),
            "transplanted_from": state_ref,
            **{key: snap[key] for key in ("time", *state.STATE_ARRAY_KEYS)},
        }
        return self.store.put("states", payload)

    # -- 实验：rollout / observe（MH3，规格 §15/§16） -------------------------

    def rollout(
        self,
        model_ref: str,
        *,
        state_ref: str | None = None,
        controller: dict[str, Any],
        duration_s: float | None = None,
        steps: int | None = None,
        seed: int = 0,
        budgets: dict[str, Any] | None = None,
    ) -> SimulationTrace:
        """有界 rollout：controller + 预算，trace 与终态不可变落盘。"""
        import mujoco

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"ROLLOUT_SEED_INVALID: {seed!r}")
        manifest = self._manifest(model_ref)
        if state_ref is not None:
            model, data = self.restore_state(model_ref, state_ref)
        else:
            spec = self._spec_from_manifest(manifest)
            model, _ = self._compile_smoke(spec)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

        plan = rollout_mod.validate_controller(controller, model.nu)
        resolved_steps = rollout_mod.resolve_steps(
            controller, steps=steps, duration_s=duration_s, timestep=float(model.opt.timestep)
        )
        states, actual_steps = rollout_mod.run_rollout(
            model, data, plan=plan, steps=resolved_steps, budgets=budgets
        )

        merged = {**rollout_mod.DEFAULT_BUDGETS, **(budgets or {})}
        digest = rollout_mod.states_digest(states)
        record = {
            "kind": "simulation_trace",
            "model_ref": model_ref,
            "model_digest": self._xml_digest(manifest),
            "seed": seed,
            "controller": controller,
            "steps": actual_steps,
            "timestep_s": float(model.opt.timestep),
            "duration_s": float(data.time),
            "states_digest": digest,
            "states": states,
        }
        if len(canonical_json(record).encode("utf-8")) > merged["max_trace_bytes"]:
            raise ValueError(f"SIM_BUDGET_EXCEEDED: trace bytes > {merged['max_trace_bytes']}")
        trace_ref = self.store.put("traces", record)
        final_state_ref = self.capture_and_store(model_ref, model, data)

        request_digest = content_hash(
            "simrol",
            {
                "model_ref": model_ref,
                "state_ref": state_ref,
                "controller": controller,
                "steps": resolved_steps,
                "seed": seed,
            },
        )
        return SimulationTrace(
            backend="mujoco",
            backend_version=manifest["backend_version"],
            created_at=self._created_at(trace_ref),
            request_digest=request_digest,
            model_ref=model_ref,
            model_digest=self._xml_digest(manifest),
            steps=actual_steps,
            timestep_s=float(model.opt.timestep),
            states_digest=digest,
            trace_ref=trace_ref,
            final_state_ref=final_state_ref,
        ).with_digest()

    def observe(self, model_ref: str, state_ref: str, channels: list[str]) -> ObservationResult:
        """语义化有界观测（规格 §16）。"""
        if not isinstance(channels, list) or not channels:
            raise ValueError("OBSERVE_CHANNELS_REQUIRED: channels must be a non-empty list")
        model, data = self.restore_state(model_ref, state_ref)
        manifest = self._manifest(model_ref)
        values = observe_mod.observe_channels(model, data, channels)
        return ObservationResult(
            backend="mujoco",
            backend_version=manifest["backend_version"],
            created_at=datetime.now(UTC).isoformat(),
            request_digest=content_hash(
                "simobq", {"model_ref": model_ref, "state_ref": state_ref, "channels": channels}
            ),
            model_ref=model_ref,
            time=float(data.time),
            values=values,
        ).with_digest()

    # -- 渲染（MH6，规格 §32） --------------------------------------------------

    def render(
        self,
        trace_ref: str,
        *,
        camera: str | None = None,
        width: int = 640,
        height: int = 480,
        max_frames: int = 16,
    ) -> dict[str, Any]:
        """把 trace 渲染成 GIF artifact 落 renders 分区。

        render 是证据 artifact，不是验证真相（规格 §32）；实际渲染后端
        诚实记录，OSMesa 不报 GPU。
        """
        if not (64 <= width <= 4096) or not (64 <= height <= 4096):
            raise ValueError(f"RENDER_INPUT_INVALID: width/height out of range: {width}x{height}")
        if not (1 <= max_frames <= 64):
            raise ValueError(f"RENDER_INPUT_INVALID: max_frames out of range: {max_frames}")
        record = self.store.get(trace_ref)
        if not isinstance(record, dict) or record.get("kind") != "simulation_trace":
            raise ValueError(f"REF_NOT_FOUND: {trace_ref!r} is not a simulation trace")
        model_ref = record["model_ref"]
        manifest = self._manifest(model_ref)
        if record.get("model_digest") != self._xml_digest(manifest):
            raise ValueError(f"CROSS_MODEL_REF: trace {trace_ref!r} does not belong to model")
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)

        import io
        import os

        import mujoco

        try:
            renderer = mujoco.Renderer(model, height, width)
        except Exception as exc:
            raise ValueError(f"SIM_RENDER_UNAVAILABLE: {type(exc).__name__}: {exc}") from exc
        try:
            from PIL import Image

            states = record["states"]
            if len(states) > max_frames:
                stride = -(-len(states) // max_frames)
                states = states[::stride]
            frames = []
            for snapshot in states:
                data = mujoco.MjData(model)
                state.apply_state(
                    model,
                    data,
                    {
                        "time": snapshot["t"],
                        "qpos": snapshot["qpos"],
                        "qvel": snapshot["qvel"],
                        "act": [0.0] * model.na,
                        "ctrl": snapshot["ctrl"],
                        "mocap_pos": [0.0] * (model.nmocap * 3),
                        "mocap_quat": [0.0] * (model.nmocap * 4),
                    },
                )
                renderer.update_scene(data, camera=camera or -1)
                frames.append(Image.fromarray(renderer.render()))
            buffer = io.BytesIO()
            frames[0].save(
                buffer, format="GIF", save_all=True, append_images=frames[1:], duration=80, loop=0
            )
            artifact_ref = self.store.put("renders", buffer.getvalue())
        finally:
            renderer.close()
        return {
            "artifact_ref": artifact_ref,
            "frames": len(frames),
            "width": width,
            "height": height,
            "camera": camera or "default",
            "renderer_backend": os.environ.get("MUJOCO_GL", "") or "default",
            "trace_ref": trace_ref,
        }

    # -- 物理诚实审计（MH4，规格 §17-§19） ------------------------------------

    def audit(
        self,
        model_ref: str,
        *,
        checks: list[str] | None = None,
        trace_ref: str | None = None,
        state_ref: str | None = None,
        policy: AuditPolicy = STRICT_POLICY,
    ) -> AuditResult:
        """物理诚实审计：A01-A08 + A15-A20，机器可读结果落 audits 分区。"""
        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)

        trace_record = None
        if trace_ref is not None:
            record = self.store.get(trace_ref)
            if not isinstance(record, dict) or record.get("kind") != "simulation_trace":
                raise ValueError(f"REF_NOT_FOUND: {trace_ref!r} is not a simulation trace")
            if record.get("model_digest") != self._xml_digest(manifest):
                raise ValueError(f"CROSS_MODEL_REF: trace {trace_ref!r} does not belong to model")
            trace_record = record

        ctx = AuditContext(
            model=model,
            spec=spec,
            xml_text=manifest["mjcf_xml"],
            policy=policy,
            trace_record=trace_record,
            state_ref=state_ref,
            restore_fn=lambda sr: self.restore_state(model_ref, sr),
        )
        outcome = run_checks(ctx, checks)

        evidence = [trace_ref] if trace_ref is not None else []
        payload = {
            "kind": "audit_result",
            "model_ref": model_ref,
            "model_digest": self._xml_digest(manifest),
            "status": outcome["status"],
            "checks": outcome["checks"],
            "violations": outcome["violations"],
            "warnings": outcome["warnings"],
            "evidence": evidence,
            "backend": "mujoco",
            "backend_version": manifest["backend_version"],
        }
        audit_ref = self.store.put("audits", payload)
        return AuditResult(
            backend="mujoco",
            backend_version=manifest["backend_version"],
            created_at=self._created_at(audit_ref),
            request_digest=content_hash(
                "simauq",
                {
                    "model_ref": model_ref,
                    "checks": checks,
                    "trace_ref": trace_ref,
                    "state_ref": state_ref,
                },
            ),
            status=outcome["status"],
            checks=outcome["checks"],
            warnings=[f"{w['check']}: {w.get('reason', 'warn')}" for w in outcome["warnings"]],
            violations=[
                f"{v['check']}: {v.get('reason', 'violation')}" for v in outcome["violations"]
            ],
            evidence=evidence,
            audit_ref=audit_ref,
            violations_detail=outcome["violations"],
            warnings_detail=outcome["warnings"],
        ).with_digest()

    # -- 实验回执 / strict replay / 对比（MH5，规格 §20/§30/§31/§56/§57） ------

    def run_experiment(
        self,
        model_ref: str,
        *,
        state_ref: str | None = None,
        controller: dict[str, Any],
        duration_s: float | None = None,
        steps: int | None = None,
        seed: int = 0,
        budgets: dict[str, Any] | None = None,
        audit: bool = True,
    ) -> SimulationReceipt:
        """rollout + 指标 + 审计 → SimulationReceipt（不可变落盘，幂等）。"""
        import mujoco

        manifest = self._manifest(model_ref)
        if state_ref is not None:
            model, data = self.restore_state(model_ref, state_ref)
        else:
            spec = self._spec_from_manifest(manifest)
            model, _ = self._compile_smoke(spec)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        initial_state_ref = self.capture_and_store(model_ref, model, data)

        plan = rollout_mod.validate_controller(controller, model.nu)
        resolved_steps = rollout_mod.resolve_steps(
            controller, steps=steps, duration_s=duration_s, timestep=float(model.opt.timestep)
        )
        tracked = [
            (i, int(model.jnt_qposadr[int(model.actuator_trnid[i][0])]))
            for i in range(model.nu)
            if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        ]
        collector = _make_collector(model, data, plan, tracked)
        states, actual_steps = rollout_mod.run_rollout(
            model, data, plan=plan, steps=resolved_steps, budgets=budgets, visit=collector.visit
        )
        merged = {**rollout_mod.DEFAULT_BUDGETS, **(budgets or {})}
        digest = rollout_mod.states_digest(states)
        record = {
            "kind": "simulation_trace",
            "model_ref": model_ref,
            "model_digest": self._xml_digest(manifest),
            "seed": seed,
            "controller": controller,
            "steps": actual_steps,
            "timestep_s": float(model.opt.timestep),
            "duration_s": float(data.time),
            "states_digest": digest,
            "states": states,
        }
        if len(canonical_json(record).encode("utf-8")) > merged["max_trace_bytes"]:
            raise ValueError(f"SIM_BUDGET_EXCEEDED: trace bytes > {merged['max_trace_bytes']}")
        trace_ref = self.store.put("traces", record)
        final_state_ref = self.capture_and_store(model_ref, model, data)

        metrics = collector.finalize(
            timestep=float(model.opt.timestep), duration_s=float(data.time)
        )
        audit_ref = ""
        success: bool | None = None
        if audit:
            audit_result = self.audit(model_ref, trace_ref=trace_ref)
            audit_ref = audit_result.audit_ref
            success = audit_result.status == "PASS"

        semantic_digest = content_hash(
            "simrcp", {"metrics": _round_metrics(metrics), "success": success}
        )
        payload = {
            "kind": "simulation_receipt",
            "schema_version": "rosclaw.sim.receipt.v1",
            "backend": "mujoco",
            "backend_version": manifest["backend_version"],
            "model_ref": model_ref,
            "model_digest": self._xml_digest(manifest),
            "initial_state_ref": initial_state_ref,
            "action_digest": content_hash("simact", controller),
            "trace_ref": trace_ref,
            "final_state_ref": final_state_ref,
            "seed": seed,
            "steps": actual_steps,
            "simulation_time_s": float(data.time),
            "success": success,
            "metrics": metrics,
            "audit_ref": audit_ref,
            "artifacts": [],
            "states_digest": digest,
            "semantic_digest": semantic_digest,
            "trust_level": "SIMULATED",
            "usable_for_real_execution": False,
        }
        receipt_ref = self.store.put("experiments", payload)
        return SimulationReceipt(
            backend="mujoco",
            backend_version=manifest["backend_version"],
            created_at=self._created_at(receipt_ref),
            model_ref=model_ref,
            model_digest=payload["model_digest"],
            initial_state_ref=initial_state_ref,
            action_digest=payload["action_digest"],
            trace_ref=trace_ref,
            final_state_ref=final_state_ref,
            seed=seed,
            steps=actual_steps,
            simulation_time_s=payload["simulation_time_s"],
            success=success,
            metrics=metrics,
            audit_ref=audit_ref,
            states_digest=digest,
            semantic_digest=semantic_digest,
            receipt_ref=receipt_ref,
        ).with_digest()

    def strict_replay(self, receipt_ref: str) -> dict[str, Any]:
        """规格 §57：重放校验 model/backend/seed/初始状态/controller/steps/
        关键指标/任务结果；不一致 → REPLAY_DIVERGED，不得 promotion。

        双层 digest（§56）：raw（states_digest 逐状态）优先；raw 不符时
        退化到语义层（指标容差 + success）判定。
        """
        import mujoco

        payload = self.store.get(receipt_ref)
        if not isinstance(payload, dict) or payload.get("kind") != "simulation_receipt":
            raise ValueError(f"REF_NOT_FOUND: {receipt_ref!r} is not a simulation receipt")
        current_version = str(mujoco.__version__)
        if payload.get("backend") != "mujoco" or payload.get("backend_version") != current_version:
            raise ValueError(
                f"REPLAY_DIVERGED: backend {payload.get('backend')}"
                f"@{payload.get('backend_version')} != mujoco@{current_version}"
            )
        manifest = self._manifest(payload["model_ref"])
        if payload["model_digest"] != self._xml_digest(manifest):
            raise ValueError("REPLAY_DIVERGED: model digest mismatch")

        trace_record = self.store.get(payload["trace_ref"])
        controller = trace_record["controller"]
        model, data = self.restore_state(payload["model_ref"], payload["initial_state_ref"])
        plan = rollout_mod.validate_controller(controller, model.nu)
        tracked = [
            (i, int(model.jnt_qposadr[int(model.actuator_trnid[i][0])]))
            for i in range(model.nu)
            if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        ]
        collector = _make_collector(model, data, plan, tracked)
        states, _ = rollout_mod.run_rollout(
            model, data, plan=plan, steps=int(payload["steps"]), visit=collector.visit
        )
        if rollout_mod.states_digest(states) == payload["states_digest"]:
            return {"verified": True, "mode": "raw", "receipt_ref": receipt_ref}

        metrics = collector.finalize(
            timestep=float(model.opt.timestep), duration_s=float(data.time)
        )
        audit_status = self.audit(payload["model_ref"], trace_ref=payload["trace_ref"]).status
        if (
            _metrics_close(metrics, payload["metrics"])
            and (audit_status == "PASS") == payload["success"]
        ):
            return {"verified": True, "mode": "semantic", "receipt_ref": receipt_ref}
        raise ValueError("REPLAY_DIVERGED: raw states and semantic metrics both mismatch")

    def compare_experiments(self, receipt_refs: list[str]) -> ComparisonResult:
        """规格 §20：指标表 + best + Pareto 候选（机器比较，不靠 LLM 读数）。"""
        if not isinstance(receipt_refs, list) or len(receipt_refs) < 2:
            raise ValueError("COMPARE_REFS_REQUIRED: need >= 2 receipt refs")
        payloads = []
        for ref in receipt_refs:
            payload = self.store.get(ref)
            if not isinstance(payload, dict) or payload.get("kind") != "simulation_receipt":
                raise ValueError(f"REF_NOT_FOUND: {ref!r} is not a simulation receipt")
            payloads.append({**payload, "_ref": ref})
        table = compare_mod.build_metric_table(payloads)
        pareto = compare_mod.pareto_front(table)
        best = compare_mod.best_experiment(table, pareto)
        return ComparisonResult(
            backend="mujoco",
            backend_version=payloads[0]["backend_version"],
            created_at=datetime.now(UTC).isoformat(),
            subject_refs=list(receipt_refs),
            metric_table=table,
            best_ref=best,
            pareto_refs=pareto,
        ).with_digest()

    # -- 内部 ---------------------------------------------------------------

    def _manifest(self, model_ref: str) -> dict[str, Any]:
        manifest = self.store.get(model_ref)  # REF_NOT_FOUND / 篡改 fail closed
        if not isinstance(manifest, dict) or manifest.get("kind") != "model_manifest":
            raise ValueError(f"MODEL_NOT_FOUND: {model_ref!r} is not a model manifest")
        return manifest

    def _load_assets(self, manifest: dict[str, Any]) -> dict[str, bytes]:
        return {name: self.store.get(ref) for name, ref in manifest["assets"].items()}

    def _spec_from_manifest(self, manifest: dict[str, Any]):  # noqa: ANN202
        import mujoco

        try:
            return mujoco.MjSpec.from_string(
                manifest["mjcf_xml"], assets=self._load_assets(manifest) or None
            )
        except Exception as exc:
            raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc

    @staticmethod
    def _compile_with_warnings(spec) -> list[str]:  # noqa: ANN001
        _, warnings = MujocoBackend._compile_smoke(spec)
        return warnings

    @staticmethod
    def _compile_smoke(spec):  # noqa: ANN001, ANN202
        """编译冒烟；编译警告 best-effort 捕获（3.11 实测多数场景静默，
        字段常驻但允许为空——见 ADR-0014 补充段）。"""
        import mujoco

        warnings: list[str] = []
        hook = getattr(mujoco, "set_mju_user_warning", None)
        hooked = False
        if hook is not None:
            try:
                hook(lambda msg: warnings.append(str(msg)))
                hooked = True
            except Exception:  # noqa: BLE001 —— best-effort
                hooked = False
        try:
            model = spec.compile()
        except Exception as exc:
            raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc
        finally:
            if hooked:
                with contextlib.suppress(Exception):
                    hook(None)
        return model, warnings

    def _capture_assets(self, spec, base_dir: Path) -> dict[str, bytes]:  # noqa: ANN001
        """捕获 file 引用的 mesh/texture 资产字节（containment 校验）。"""
        assets: dict[str, bytes] = {}
        for dirname, elements in (
            (getattr(spec, "meshdir", "") or "", getattr(spec, "meshes", [])),
            (getattr(spec, "texturedir", "") or "", getattr(spec, "textures", [])),
        ):
            for element in elements:
                filename = getattr(element, "file", "") or ""
                if not filename:
                    continue
                key = posixpath.join(dirname, filename) if dirname else filename
                source = (base_dir / key).resolve()
                if base_dir.resolve() not in source.parents and source != base_dir.resolve():
                    raise ValueError(f"MODEL_PATH_ESCAPE: asset {key!r} escapes model directory")
                if not source.is_file():
                    raise ValueError(f"MODEL_NOT_FOUND: asset {key!r}")
                assets[key] = source.read_bytes()
        return assets

    @staticmethod
    def _xml_digest(manifest: dict[str, Any]) -> str:
        return "sha256:" + hashlib.sha256(manifest["mjcf_xml"].encode("utf-8")).hexdigest()

    def _created_at(self, ref: str) -> str:
        """不可变 manifest 的落盘 mtime = 首次创建时间（幂等稳定）。"""
        mtime = self.store.resolve(ref).stat().st_mtime
        return datetime.fromtimestamp(mtime, UTC).isoformat()


def _make_collector(model, data, plan: dict[str, Any], tracked):  # noqa: ANN001, ANN202
    """构造指标采集器：position_targets 先应用（run_rollout 内幂等再
    应用一次）；ctrl_series 逐行目标由 collector 查表。"""
    if plan["kind"] == "position_targets":
        for i, v in enumerate(plan["values"]):
            data.ctrl[i] = v
    return MetricCollector(model, data, tracked, series_rows=plan.get("rows"))


def _round_metrics(metrics: dict[str, Any], digits: int = 6) -> dict[str, Any]:
    """语义 digest 用：浮点指标舍入（规格 §56 canonical tolerance）。"""
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            rounded[key] = value
        elif isinstance(value, (int, float)):
            rounded[key] = round(float(value), digits)
        else:
            rounded[key] = value
    return rounded


def _metrics_close(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """语义层指标容差比较：bool 精确，数值 rel=1e-6/abs=1e-9。"""
    import math

    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, bool):
            if actual_value != expected_value:
                return False
        elif isinstance(expected_value, (int, float)):
            if not isinstance(actual_value, (int, float)) or not math.isclose(
                float(actual_value), float(expected_value), rel_tol=1e-6, abs_tol=1e-9
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True
