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
    ModelInspection,
    ModelPatchResult,
    ModelReference,
    ObservationResult,
    SimulationBackendCapabilities,
    SimulationTrace,
)
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
        import mujoco

        path = resolve_mjcf_source(asset_ref, task_root=self._task_root)
        xml_bytes = path.read_bytes()

        try:
            spec = mujoco.MjSpec.from_file(str(path))
        except Exception as exc:
            raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc
        warnings = self._compile_with_warnings(spec)  # 编译冒烟（失败即拒）

        assets: dict[str, str] = {}
        for key, data in self._capture_assets(spec, path.parent).items():
            assets[key] = self.store.put("models", data)

        manifest = {
            "kind": "model_manifest",
            "mjcf_xml": xml_bytes.decode("utf-8"),
            "assets": assets,
            "source": {
                "kind": source_kind_for(path, task_root=self._task_root),
                "ref": str(asset_ref),
            },
            "parent_model_ref": None,
            "patches": [],
            "compile_warnings": warnings,
            "backend": "mujoco",
            "backend_version": str(mujoco.__version__),
        }
        ref = self.store.put("models", manifest)

        return ModelReference(
            backend="mujoco",
            backend_version=str(mujoco.__version__),
            created_at=self._created_at(ref),
            model_ref=ref,
            model_digest="sha256:" + hashlib.sha256(xml_bytes).hexdigest(),
            source=manifest["source"],
            compiled=True,
            body_description=inspect_mjcf(path).to_dict(),
            compile_warnings=warnings,
            runtime_capabilities=self.capabilities().to_canonical_dict(),
        )

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
