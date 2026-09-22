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
from rosclaw.sim.backends.mujoco import batch as batch_mod
from rosclaw.sim.backends.mujoco import interact as interact_mod
from rosclaw.sim.backends.mujoco import observe as observe_mod
from rosclaw.sim.backends.mujoco import rollout as rollout_mod
from rosclaw.sim.backends.mujoco import state, state_v2
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
from rosclaw.sim.experiment.predicates import evaluate_predicates
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
        import mujoco  # noqa: F401 —— backend_version 需要

        spec = _spec_from_xml_assets(xml_text, assets or {})
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
            model_digest=self._model_digest(manifest),
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
        detail = inspect_model_full(model, model_digest=self._model_digest(manifest), spec=spec)
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
            model_digest=self._model_digest(manifest),
            parent_model_ref=manifest["parent_model_ref"],
            source=manifest["source"],
            compiled=True,
        )

    def patch_model(self, model_ref: str, patches: list[dict[str, Any]]) -> ModelPatchResult:
        """MjSpec 结构化补丁：apply → compile → 新 manifest（母模型不动）。"""

        manifest = self._manifest(model_ref)
        spec = _spec_from_xml_assets(manifest["mjcf_xml"], self._load_assets(manifest))
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
        if values.get("model_digest") != self._model_digest(manifest):
            raise ValueError(
                f"CROSS_MODEL_REF: state digest {values.get('model_digest')!r} != model digest"
            )
        state.validate_state_values(model, values)
        payload = {
            "kind": "state_snapshot",
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
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
        values["model_digest"] = self._model_digest(manifest)
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
        if snap.get("model_digest") != self._model_digest(manifest):
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
        if not isinstance(snap, dict) or snap.get("kind") not in (
            "state_snapshot",
            "state_snapshot_v2",
        ):
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")
        if snap.get("model_digest") != self._model_digest(manifest):
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

        与 restore 的 fail-closed 不同，这是**显式操作**，双重校验
        （0915 §八）：
        1. 维度完全一致（nq/nv/na/nu/nmocap），否则 STATE_DIMENSION；
        2. **结构签名一致**（joint 名/类型/qpos 地址/dof 地址/actuator→
           joint 映射/mocap 布局）——维度相同但 shoulder↔wrist 语义
           不同的模型拒绝 STATE_INCOMPATIBLE。
        参数 patch（kp/damping/friction/mass）不改签名 → 允许。
        移植记录 provenance（transplanted_from）。
        """
        manifest = self._manifest(model_ref)
        snap = self.store.get(state_ref)
        if not isinstance(snap, dict) or snap.get("kind") not in (
            "state_snapshot",
            "state_snapshot_v2",
        ):
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        source_ref = snap.get("model_ref")
        if source_ref:
            source_manifest = self._manifest(source_ref)
            source_spec = self._spec_from_manifest(source_manifest)
            source_model, _ = self._compile_smoke(source_spec)
            if self._structural_signature(source_model) != self._structural_signature(model):
                raise ValueError(
                    f"STATE_INCOMPATIBLE: structural signature changed "
                    f"({source_ref!r} → {model_ref!r})"
                )
        if snap.get("kind") == "state_snapshot_v2":
            # v2：结构签名一致 ⇒ 状态布局一致；mj_setState 再做尺寸校验。
            import mujoco as _mujoco
            import numpy as np

            blob = self.store.get(snap["state_vector_ref"])
            if not isinstance(blob, bytes):
                raise ValueError(
                    f"STORE_DIGEST_MISMATCH: state vector {snap['state_vector_ref']!r}"
                )
            vector = np.frombuffer(blob, dtype=np.float64).copy()
            state_v2.apply_state_v2(model, _mujoco.MjData(model), vector, snap["state_spec_value"])
            meta = {
                **snap,
                "model_ref": model_ref,
                "model_digest": self._model_digest(manifest),
                "transplanted_from": state_ref,
            }
            return self.store.put("states", meta)
        state.validate_state_values(model, snap)
        payload = {
            "kind": "state_snapshot",
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
            "transplanted_from": state_ref,
            **{key: snap[key] for key in ("time", *state.STATE_ARRAY_KEYS)},
        }
        return self.store.put("states", payload)

    @staticmethod
    def _structural_signature(model) -> dict[str, Any]:  # noqa: ANN001
        """joint 名/类型/qpos 地址/dof 地址/actuator→joint 映射/mocap 布局。"""
        import mujoco

        return {
            "joints": [
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}",
                    int(model.jnt_type[i]),
                    int(model.jnt_qposadr[i]),
                    int(model.jnt_dofadr[i]),
                )
                for i in range(model.njnt)
            ],
            "actuators": [
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}",
                    int(model.actuator_trnid[i][0]),
                )
                for i in range(model.actuator_trnid.shape[0])
            ],
            "nmocap": int(model.nmocap),
        }

    def _control_schema(self, model, manifest: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ANN001
        """ctrl 通道语义（MH10）：从来源 MjSpec 推导。"""
        from rosclaw.sim.backends.mujoco.inspect import _control_channels

        spec = self._spec_from_manifest(manifest)
        return _control_channels(model, spec)

    # -- 便携模型工件（MH10b，0916 §4.3） -------------------------------------

    def export_model_mjz(self, model_ref: str) -> dict[str, Any]:
        """导出 .mjz 便携模型工件（spec.assets 填充 + to_zip，
        from_zip 自包含编译——跨机器证据/Hub/benchmark/replay 用）。"""
        import tempfile

        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        spec.assets = self._load_assets(manifest)
        with tempfile.NamedTemporaryFile(suffix=".mjz", delete=False) as tmp:
            spec.to_zip(tmp.name)
            blob = Path(tmp.name).read_bytes()
        artifact_ref = self.store.put("models", blob)
        return {
            "artifact_ref": artifact_ref,
            "format": "mjz",
            "size_bytes": len(blob),
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
        }

    # -- 状态 v2：mjSTATE_INTEGRATION 真值（MH10，0916 §二） --------------------

    def state_fidelity(self, state_ref: str) -> str:
        """快照保真度：FULL_INTEGRATION / FULL_PHYSICS / LEGACY_PARTIAL。"""
        snap = self.store.get(state_ref)
        if not isinstance(snap, dict):
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r}")
        if snap.get("kind") == "state_snapshot_v2":
            return snap["fidelity"]
        if snap.get("kind") == "state_snapshot":
            return state_v2.FIDELITY_LEGACY_PARTIAL
        raise ValueError(f"REF_NOT_FOUND: {state_ref!r} is not a state snapshot")

    def capture_and_store_v2(
        self,
        model_ref: str,
        model,  # noqa: ANN001
        data,  # noqa: ANN001
        *,
        fidelity: str = state_v2.FIDELITY_FULL_INTEGRATION,
    ) -> str:
        """mj_getState 捕获 + 元数据 JSON 与 float64 blob 分离落盘。"""
        manifest = self._manifest(model_ref)
        vector, spec_value = state_v2.capture_state_v2(model, data, fidelity=fidelity)
        blob = vector.tobytes()
        blob_ref = self.store.put("states", blob)
        meta = {
            "kind": "state_snapshot_v2",
            "schema_version": "rosclaw.sim.state.v2",
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
            "state_spec": state_v2.spec_name(spec_value),
            "state_spec_value": spec_value,
            "state_size": len(vector),
            "state_vector_ref": blob_ref,
            "state_digest": "sha256:" + hashlib.sha256(blob).hexdigest(),
            "structural_signature": self._structural_signature_str(model),
            "fidelity": fidelity,
            # preview（小数组便利字段；真值在 state_vector_ref blob）
            "time": float(data.time),
            "qpos": [float(v) for v in data.qpos],
            "qvel": [float(v) for v in data.qvel],
            "ctrl": [float(v) for v in data.ctrl],
        }
        return self.store.put("states", meta)

    def initial_state_v2(
        self,
        model_ref: str,
        *,
        fidelity: str = state_v2.FIDELITY_FULL_INTEGRATION,
    ) -> str:
        """模型的 v2 初始状态（qpos0 + mj_forward）快照，确定性幂等。"""
        import mujoco

        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return self.capture_and_store_v2(model_ref, model, data, fidelity=fidelity)

    def restore_state_v2(self, model_ref: str, state_ref: str):  # noqa: ANN202
        """v2 恢复（v1 快照走 legacy 路径，fidelity 由 state_fidelity 判定）。"""
        import numpy as np

        snap = self.store.get(state_ref)
        if not isinstance(snap, dict):
            raise ValueError(f"REF_NOT_FOUND: {state_ref!r}")
        if snap.get("kind") != "state_snapshot_v2":
            return self.restore_state(model_ref, state_ref)  # LEGACY_PARTIAL 路径
        manifest = self._manifest(model_ref)
        if snap["model_digest"] != self._model_digest(manifest):
            raise ValueError(
                f"CROSS_MODEL_REF: state {state_ref!r} does not belong to {model_ref!r}"
            )
        blob = self.store.get(snap["state_vector_ref"])
        if not isinstance(blob, bytes):
            raise ValueError(f"STORE_DIGEST_MISMATCH: state vector {snap['state_vector_ref']!r}")
        vector = np.frombuffer(blob, dtype=np.float64).copy()
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        import mujoco

        data = mujoco.MjData(model)
        state_v2.apply_state_v2(model, data, vector, snap["state_spec_value"])
        return model, data

    @staticmethod
    def _structural_signature_str(model) -> str:  # noqa: ANN001
        return content_hash("simsig", MujocoBackend._structural_signature(model))

    # -- 并行批次 rollout（MH14，0916 §十七-§十九） ----------------------------

    def rollout_batch(
        self,
        model_refs: list[str],
        *,
        controller: dict[str, Any],
        state_refs: list[str] | None = None,
        duration_s: float | None = None,
        steps: int | None = None,
        budgets: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """homogeneous 模型批量 rollout（mujoco.rollout 原生多线程）。

        CPU batch 优先于 MJX：与权威 CPU truth 同域。返回每分支
        {trace_ref, final_state_ref, states_digest, steps, execution}。

        MH20-A：state_refs（每分支 transplant 后的 v2 状态）成为正式
        接口——并行与串行同一实验起点；INTEGRATION 尾部（USER/
        WARMSTART，含 eq_active）非零即 BATCH_STATE_FIDELITY_REQUIRED
        诚实回退串行，绝不为并行降低 state fidelity。
        """
        import numpy as np

        if not isinstance(model_refs, list) or not model_refs:
            raise ValueError("BATCH_EMPTY: model_refs must be a non-empty list")
        if state_refs is not None and len(state_refs) != len(model_refs):
            raise ValueError(
                f"BATCH_STATE_COUNT_MISMATCH: {len(state_refs)} states != {len(model_refs)} models"
            )
        manifests = [self._manifest(ref) for ref in model_refs]
        models = []
        for manifest in manifests:
            spec = self._spec_from_manifest(manifest)
            model, _ = self._compile_smoke(spec)
            models.append(model)
        batch_mod.check_homogeneous(models)  # BATCH_NOT_HOMOGENEOUS / BATCH_SEMANTICS_INCOMPATIBLE

        import mujoco

        initial = None
        if state_refs is not None:
            fp_size = mujoco.mj_stateSize(models[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            vectors = []
            for ref, state_ref in zip(model_refs, state_refs, strict=True):
                meta = self.store.get(state_ref)
                blob = self.store.get(meta["state_vector_ref"])
                vector = np.frombuffer(blob, dtype=np.float64)
                # MH20-A 保真规则：eq_active 是物理状态（约束激活），
                # native batch 的 FULLPHYSICS 初值无法承载 → 诚实串行
                # 回退。warmstart/ctrl/sensordata/qacc 是求解脚手架或
                # 派生量（实测对轨迹零影响），不算物理保真损失。
                _, state_data = self.restore_state_v2(ref, state_ref)
                if np.any(np.asarray(state_data.eq_active, dtype=int) != 0):
                    raise ValueError(
                        "BATCH_STATE_FIDELITY_REQUIRED: state 含 eq_active=1"
                        "（约束激活态 ∈ INTEGRATION-only）——native batch 无法无损承载"
                    )
                if len(vector) < fp_size:
                    raise ValueError(
                        f"BATCH_STATE_DIMENSION: state vector {len(vector)} < FULLPHYSICS {fp_size}"
                    )
                vectors.append(vector[:fp_size].copy())
            initial = batch_mod.initial_vectors(models, vectors)

        plan = rollout_mod.validate_controller(
            controller, models[0].nu, channels=self._control_schema(models[0], manifests[0])
        )
        resolved_steps = rollout_mod.resolve_steps(
            controller, steps=steps, duration_s=duration_s, timestep=float(models[0].opt.timestep)
        )
        merged = {**rollout_mod.DEFAULT_BUDGETS, **(budgets or {})}
        if resolved_steps > merged["max_steps"]:
            raise ValueError(f"SIM_BUDGET_EXCEEDED: steps {resolved_steps} > {merged['max_steps']}")

        if plan["kind"] == "ctrl_series":
            ctrl_rows = np.asarray(plan["rows"], dtype=float)
        else:
            row = (
                np.zeros(models[0].nu)
                if plan["kind"] == "hold"
                else np.asarray(plan["values"], dtype=float)
            )
            ctrl_rows = np.tile(row, (resolved_steps, 1))

        state_trajs, _ = batch_mod.run_batch(models, ctrl_rows=ctrl_rows, initial=initial)
        stride = max(1, -(-resolved_steps // merged["max_record_points"]))

        results = []
        for index, (ref, manifest, model, full_traj) in enumerate(
            zip(model_refs, manifests, models, state_trajs, strict=True)
        ):
            states = batch_mod.trajectory_to_states(
                model, full_traj[::stride], ctrl_rows, record_stride=stride
            )
            # 与串行记录对齐：前置初始状态行（rollout 轨迹只含步后状态）。
            data0 = initial[index] if initial is not None else batch_mod.initial_vectors([model])[0]
            states.insert(
                0,
                {
                    "t": float(data0[0]),
                    "qpos": [float(v) for v in data0[1 : 1 + model.nq]],
                    "qvel": [float(v) for v in data0[1 + model.nq : 1 + model.nq + model.nv]],
                    "ctrl": [float(v) for v in ctrl_rows[0]],
                },
            )
            digest = rollout_mod.states_digest(states)
            record = {
                "kind": "simulation_trace",
                "model_ref": ref,
                "model_digest": self._model_digest(manifest),
                "seed": 0,
                "controller": controller,
                "steps": resolved_steps,
                "timestep_s": float(model.opt.timestep),
                "duration_s": float(full_traj[-1][0]),
                "states_digest": digest,
                "states": states,
            }
            if len(canonical_json(record).encode("utf-8")) > merged["max_trace_bytes"]:
                raise ValueError(f"SIM_BUDGET_EXCEEDED: trace bytes > {merged['max_trace_bytes']}")
            trace_ref = self.store.put("traces", record)

            import mujoco

            data = mujoco.MjData(model)
            mujoco.mj_setState(
                model,
                data,
                np.asarray(full_traj[-1], dtype=np.float64),
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            mujoco.mj_forward(model, data)
            final_state_ref = self.capture_and_store_v2(ref, model, data)
            results.append(
                {
                    "model_ref": ref,
                    "trace_ref": trace_ref,
                    "final_state_ref": final_state_ref,
                    "states_digest": digest,
                    "steps": resolved_steps,
                    "execution": "batch_parallel",
                }
            )
        return results

    # -- 可执行交互（MH12，0916 §十二-§十四） ----------------------------------

    def interact(
        self,
        model_ref: str,
        state_ref: str,
        interaction: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行一个 typed interaction：executor → 新 state_ref + receipt。

        interaction: {"executor": <注册表名>, "target": {"type","name"},
        可选 "id"/"weld"}。executor 只允许官方注册表
        （interact.EXECUTORS），不允许 arbitrary Python。
        """
        if not isinstance(interaction, dict):
            raise ValueError("INTERACTION_INVALID: interaction must be a mapping")
        executor = interaction.get("executor")
        if executor not in interact_mod.EXECUTORS:
            raise ValueError(
                f"INTERACTION_EXECUTOR_UNKNOWN: {executor!r} "
                f"(registered: {list(interact_mod.EXECUTORS)})"
            )
        target = interaction.get("target")
        if not isinstance(target, dict) or not isinstance(target.get("name"), str):
            raise ValueError("INTERACTION_INVALID: target.name must be a non-empty string")
        payload = payload or {}
        if not isinstance(payload, dict):
            raise ValueError("INTERACTION_PAYLOAD_INVALID: payload must be a mapping")

        manifest = self._manifest(model_ref)
        model, data = self.restore_state_v2(model_ref, state_ref)
        interaction = {**interaction, "_model_ref": model_ref}

        if executor == "joint_target":
            outcome = interact_mod.exec_joint_target(self, model, data, interaction, payload)
        elif executor == "actuator_setpoint":
            outcome = interact_mod.exec_actuator_setpoint(self, model, data, interaction, payload)
        elif executor == "gripper_close":
            outcome = interact_mod.exec_gripper_motion(
                self, model, data, interaction, payload, close=True
            )
        elif executor == "gripper_open":
            outcome = interact_mod.exec_gripper_motion(
                self, model, data, interaction, payload, close=False
            )
        elif executor == "constraint_attach":
            outcome = interact_mod.exec_constraint_attach(self, model, data, interaction, payload)
        else:
            outcome = interact_mod.exec_constraint_release(self, model, data, interaction, payload)

        new_state_ref = self.capture_and_store_v2(model_ref, model, data)
        receipt = {
            "kind": "interaction_receipt",
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
            "interaction_id": interaction.get("id", executor),
            "executor": executor,
            "target": target,
            "payload": payload,
            "outcome": outcome,
            "initial_state_ref": state_ref,
            "final_state_ref": new_state_ref,
            "constraint_assisted_grasp": bool(outcome.get("constraint_assisted_grasp", False)),
            "trust_level": "SIMULATED",
            "usable_for_real_execution": False,
        }
        receipt_ref = self.store.put("experiments", receipt)
        return {
            "ok": True,
            "interaction_id": receipt["interaction_id"],
            "executor": executor,
            "outcome": outcome,
            "state_ref": new_state_ref,
            "receipt_ref": receipt_ref,
            "constraint_assisted_grasp": receipt["constraint_assisted_grasp"],
            "trust_level": "SIMULATED",
            "usable_for_real_execution": False,
        }

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
            model, data = self.restore_state_v2(model_ref, state_ref)
        else:
            spec = self._spec_from_manifest(manifest)
            model, _ = self._compile_smoke(spec)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

        plan = rollout_mod.validate_controller(
            controller, model.nu, channels=self._control_schema(model, manifest)
        )
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
            "model_digest": self._model_digest(manifest),
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
        final_state_ref = self.capture_and_store_v2(model_ref, model, data)

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
            model_digest=self._model_digest(manifest),
            steps=actual_steps,
            timestep_s=float(model.opt.timestep),
            states_digest=digest,
            trace_ref=trace_ref,
            final_state_ref=final_state_ref,
        ).with_digest()

    def observe(self, model_ref: str, state_ref: str, channels: list[str]) -> ObservationResult:
        """语义化有界观测（规格 §16 + 0916 §十六多模态）。

        物理通道走 in-process 计算；camera_rgb/depth/segmentation 走
        隔离子进程渲染，返回 artifact_ref + intrinsics/extrinsics——
        不把图像数组塞进 JSON。
        """
        if not isinstance(channels, list) or not channels:
            raise ValueError("OBSERVE_CHANNELS_REQUIRED: channels must be a non-empty list")
        model, data = self.restore_state_v2(model_ref, state_ref)
        manifest = self._manifest(model_ref)

        camera_channels = [
            c
            for c in channels
            if c.startswith(("camera_rgb:", "camera_depth:", "camera_segmentation:"))
        ]
        physics_channels = [c for c in channels if c not in camera_channels]
        values = (
            observe_mod.observe_channels(model, data, physics_channels) if physics_channels else {}
        )
        for channel in camera_channels:
            values[channel] = self._observe_camera(manifest, model, data, channel)

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

    def _observe_camera(
        self, manifest: dict[str, Any], model, data, channel: str
    ) -> dict[str, Any]:  # noqa: ANN001
        """渲染单相机通道为 PNG artifact（隔离子进程）。"""
        import math

        import mujoco

        kind, _, camera_name = channel.partition(":")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"OBSERVE_CHANNEL_UNKNOWN: {channel}")

        width, height = 640, 480
        fovy = float(model.cam_fovy[camera_id])
        intrinsics = {
            "fovy_deg": fovy,
            "focal_px": height / (2 * math.tan(math.radians(fovy) / 2)),
            "principal_point": [width / 2, height / 2],
        }
        extrinsics = {
            "pos": [float(v) for v in data.cam_xpos[camera_id]],
            "mat": [float(v) for v in data.cam_xmat[camera_id]],
        }

        gif_backend, artifact_ref, dtype = self._render_camera_frame_subprocess(
            manifest, model, data, kind=kind, camera_name=camera_name, width=width, height=height
        )
        return {
            "artifact_ref": artifact_ref,
            "width": width,
            "height": height,
            "dtype": dtype,
            "camera": camera_name,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "renderer_backend": gif_backend,
            "simulation_time": float(data.time),
        }

    @staticmethod
    def _camera_details(model) -> list[dict[str, Any]]:  # noqa: ANN001
        import mujoco

        return [
            {"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera_{i}"}
            for i in range(model.ncam)
        ]

    def _render_camera_frame_subprocess(
        self,
        manifest: dict[str, Any],
        model,  # noqa: ANN001
        data,  # noqa: ANN001
        *,
        kind: str,
        camera_name: str,
        width: int,
        height: int,
    ) -> tuple[str, str, str]:
        """单帧相机渲染（隔离子进程，egl→osmesa，绝不走 auto）。"""
        import json
        import subprocess
        import sys
        import tempfile

        dtype_map = {
            "camera_rgb": "uint8",
            "camera_depth": "uint16",
            "camera_segmentation": "uint8",
        }
        with tempfile.TemporaryDirectory(prefix="rosclaw_cam_") as tmp:
            tmp_path = Path(tmp)
            model_file = tmp_path / "model.xml"
            model_file.write_text(manifest["mjcf_xml"], encoding="utf-8")
            for name, blob in self._load_assets(manifest).items():
                target = tmp_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
            request = {
                "model_path": str(model_file),
                "state": {
                    "time": float(data.time),
                    "qpos": [float(v) for v in data.qpos],
                    "qvel": [float(v) for v in data.qvel],
                    "ctrl": [float(v) for v in data.ctrl],
                },
                "camera": camera_name,
                "kind": kind,
                "width": width,
                "height": height,
                "out": str(tmp_path / "frame.png"),
            }
            request_file = tmp_path / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")

            errors = []
            for candidate in ("egl", "osmesa"):
                proc = subprocess.run(
                    [sys.executable, "-c", _CAMERA_WORKER_CODE, candidate, str(request_file)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode == 0 and (tmp_path / "frame.png").is_file():
                    artifact_ref = self.store.put("renders", (tmp_path / "frame.png").read_bytes())
                    return candidate, artifact_ref, dtype_map[kind]
                tail = (proc.stderr or "").strip().splitlines()
                errors.append(f"{candidate}: {tail[-1] if tail else 'native crash'}")
            raise ValueError("SIM_RENDER_UNAVAILABLE: " + "; ".join(errors)) from None

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
        if record.get("model_digest") != self._model_digest(manifest):
            raise ValueError(f"CROSS_MODEL_REF: trace {trace_ref!r} does not belong to model")
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)

        states = record["states"]
        if len(states) > max_frames:
            stride = -(-len(states) // max_frames)
            states = states[::stride]

        gif_bytes, actual_backend = self._render_gif_subprocess(
            manifest, states, camera=camera, width=width, height=height
        )
        artifact_ref = self.store.put("renders", gif_bytes)
        return {
            "artifact_ref": artifact_ref,
            "frames": len(states),
            "width": width,
            "height": height,
            "camera": camera or "default",
            "renderer_backend": actual_backend,
            "trace_ref": trace_ref,
        }

    def _render_gif_subprocess(
        self,
        manifest: dict[str, Any],
        states: list[dict[str, Any]],
        *,
        camera: str | None,
        width: int,
        height: int,
    ) -> tuple[bytes, str]:
        """整个渲染在**隔离子进程**完成（WP3 实证纪律：GL 上下文创建
        在宿主进程内可能 native abort——Jetson 上 egl/osmesa/glfw
        三类后端的初始化崩溃都实测复现过）。

        后端 egl → osmesa 逐个尝试，绝不走 auto（auto 会选 glfw——
        递归初始化崩 libc++abi）。返回 (GIF 字节, 实际使用的后端)——
        记录真实后端，不是环境声明（0915 §十一）。
        """
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory(prefix="rosclaw_render_") as tmp:
            tmp_path = Path(tmp)
            model_file = tmp_path / "model.xml"
            model_file.write_text(manifest["mjcf_xml"], encoding="utf-8")
            for name, blob_ref in manifest["assets"].items():
                target = tmp_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.store.get(blob_ref))
            request = {
                "model_path": str(model_file),
                "states": states,
                "camera": camera,
                "width": width,
                "height": height,
                "out": str(tmp_path / "out.gif"),
            }
            request_file = tmp_path / "request.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")

            errors = []
            for candidate in ("egl", "osmesa"):
                proc = subprocess.run(
                    [sys.executable, "-c", _RENDER_WORKER_CODE, candidate, str(request_file)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc.returncode == 0 and (tmp_path / "out.gif").is_file():
                    return (tmp_path / "out.gif").read_bytes(), candidate
                tail = (proc.stderr or "").strip().splitlines()
                errors.append(f"{candidate}: {tail[-1] if tail else 'native crash'}")
            raise ValueError("SIM_RENDER_UNAVAILABLE: " + "; ".join(errors)) from None

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
            if record.get("model_digest") != self._model_digest(manifest):
                raise ValueError(f"CROSS_MODEL_REF: trace {trace_ref!r} does not belong to model")
            trace_record = record

        ctx = AuditContext(
            model=model,
            spec=spec,
            xml_text=manifest["mjcf_xml"],
            policy=policy,
            trace_record=trace_record,
            state_ref=state_ref,
            restore_fn=lambda sr: self.restore_state_v2(model_ref, sr),
        )
        outcome = run_checks(ctx, checks)

        evidence = [trace_ref] if trace_ref is not None else []
        payload = {
            "kind": "audit_result",
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
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
        task_predicates: list[dict[str, Any]] | None = None,
    ) -> SimulationReceipt:
        """rollout + 指标 + 审计 + 任务谓词 → SimulationReceipt（幂等）。

        成功语义三分（0915 优化 §三）：simulation_valid（跑完）/
        physical_audit_pass（审计过）/ task_success（谓词机器判定）；
        verification_status ∈ PASS / FAIL / NOT_EVALUATED。
        """
        import mujoco

        manifest = self._manifest(model_ref)
        if state_ref is not None:
            model, data = self.restore_state_v2(model_ref, state_ref)
        else:
            spec = self._spec_from_manifest(manifest)
            model, _ = self._compile_smoke(spec)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
        initial_state_ref = self.capture_and_store_v2(model_ref, model, data)

        plan = rollout_mod.validate_controller(
            controller, model.nu, channels=self._control_schema(model, manifest)
        )
        resolved_steps = rollout_mod.resolve_steps(
            controller, steps=steps, duration_s=duration_s, timestep=float(model.opt.timestep)
        )
        tracked = [
            (i, int(model.jnt_qposadr[int(model.actuator_trnid[i][0])]))
            # nu 是 ctrl 维不是执行器个数（MH10 实证；PID 多槽会越界——
            # B05 复现）。
            for i in range(model.actuator_trnid.shape[0])
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
            "model_digest": self._model_digest(manifest),
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
        final_state_ref = self.capture_and_store_v2(model_ref, model, data)

        metrics = collector.finalize(
            timestep=float(model.opt.timestep), duration_s=float(data.time)
        )

        simulation_valid = True  # 发散在 run_rollout 已 fail closed
        physical_audit_pass: bool | None = None
        audit_ref = ""
        if audit:
            audit_result = self.audit(model_ref, trace_ref=trace_ref)
            audit_ref = audit_result.audit_ref
            physical_audit_pass = audit_result.status == "PASS"

        task_success: bool | None = None
        if task_predicates is not None:
            channels = sorted({p["channel"] for p in task_predicates})
            observations = self.observe(model_ref, final_state_ref, channels).values
            verdicts = evaluate_predicates(observations, task_predicates)
            task_success = all(v["ok"] for v in verdicts)

        if physical_audit_pass is False:
            verification_status = "FAIL"
        elif task_predicates is not None:
            verification_status = "PASS" if task_success else "FAIL"
        else:
            verification_status = "NOT_EVALUATED"

        semantic_digest = content_hash(
            "simrcp",
            {
                "metrics": _round_metrics(metrics),
                "verification_status": verification_status,
                "task_success": task_success,
            },
        )
        payload = {
            "kind": "simulation_receipt",
            "schema_version": "rosclaw.sim.receipt.v1",
            "backend": "mujoco",
            "backend_version": manifest["backend_version"],
            "model_ref": model_ref,
            "model_digest": self._model_digest(manifest),
            "initial_state_ref": initial_state_ref,
            "action_digest": content_hash("simact", controller),
            "trace_ref": trace_ref,
            "final_state_ref": final_state_ref,
            "seed": seed,
            "steps": actual_steps,
            "simulation_time_s": float(data.time),
            "success": task_success,  # 兼容字段 ≡ task_success（0915 §三）
            "simulation_valid": simulation_valid,
            "physical_audit_pass": physical_audit_pass,
            "task_success": task_success,
            "task_predicates": task_predicates,
            "verification_status": verification_status,
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
            success=task_success,
            simulation_valid=simulation_valid,
            physical_audit_pass=physical_audit_pass,
            task_success=task_success,
            verification_status=verification_status,
            metrics=metrics,
            audit_ref=audit_ref,
            states_digest=digest,
            semantic_digest=semantic_digest,
            receipt_ref=receipt_ref,
        ).with_digest()

    def strict_replay(self, receipt_ref: str) -> dict[str, Any]:
        """规格 §57 + 0915 §十：重放校验 model/backend/seed/初始状态/
        controller/steps/关键指标/任务判定；错误分类精确：
        REPLAY_ENV_MISMATCH / REPLAY_MODEL_MISMATCH /
        REPLAY_STATE_MISMATCH / REPLAY_PHYSICS_DIVERGED。

        双层 digest（§56）：raw 优先；raw 不符退化到语义层
        （指标容差 + verification_status 复算）判定。
        """
        import mujoco

        payload = self.store.get(receipt_ref)
        if not isinstance(payload, dict) or payload.get("kind") != "simulation_receipt":
            raise ValueError(f"REF_NOT_FOUND: {receipt_ref!r} is not a simulation receipt")
        current_version = str(mujoco.__version__)
        if payload.get("backend") != "mujoco" or payload.get("backend_version") != current_version:
            raise ValueError(
                f"REPLAY_ENV_MISMATCH: backend {payload.get('backend')}"
                f"@{payload.get('backend_version')} != mujoco@{current_version}"
            )
        manifest = self._manifest(payload["model_ref"])
        if payload["model_digest"] != self._model_digest(manifest):
            raise ValueError("REPLAY_MODEL_MISMATCH: model digest mismatch")

        trace_record = self.store.get(payload["trace_ref"])
        if not isinstance(trace_record, dict):
            raise ValueError(f"REF_NOT_FOUND: {payload['trace_ref']!r} is not a simulation trace")
        controller = trace_record["controller"]
        fidelity = self.state_fidelity(payload["initial_state_ref"])
        try:
            model, data = self.restore_state_v2(payload["model_ref"], payload["initial_state_ref"])
        except ValueError as exc:
            raise ValueError(f"REPLAY_STATE_MISMATCH: {exc}") from exc
        plan = rollout_mod.validate_controller(
            controller, model.nu, channels=self._control_schema(model, manifest)
        )
        tracked = [
            (i, int(model.jnt_qposadr[int(model.actuator_trnid[i][0])]))
            # nu 是 ctrl 维不是执行器个数（MH10 实证；PID 多槽会越界——
            # B05 复现）。
            for i in range(model.actuator_trnid.shape[0])
            if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        ]
        collector = _make_collector(model, data, plan, tracked)
        states, _ = rollout_mod.run_rollout(
            model, data, plan=plan, steps=int(payload["steps"]), visit=collector.visit
        )
        if rollout_mod.states_digest(states) == payload["states_digest"]:
            # 0916 §二.5：只有 FULL_INTEGRATION 状态允许 RAW_EXACT；
            # LEGACY_PARTIAL 最多 SEMANTIC——旧证据不升级为强证据。
            mode = "RAW_EXACT" if fidelity == state_v2.FIDELITY_FULL_INTEGRATION else "SEMANTIC"
            return {
                "verified": True,
                "mode": mode,
                "state_fidelity": fidelity,
                "receipt_ref": receipt_ref,
            }

        metrics = collector.finalize(
            timestep=float(model.opt.timestep), duration_s=float(data.time)
        )
        verification_status = self._replay_verification_status(
            model_ref=payload["model_ref"], model=model, data=data, payload=payload
        )
        if _metrics_close(metrics, payload["metrics"]) and verification_status == payload.get(
            "verification_status", "NOT_EVALUATED"
        ):
            return {
                "verified": True,
                "mode": "SEMANTIC",
                "state_fidelity": fidelity,
                "receipt_ref": receipt_ref,
            }
        raise ValueError("REPLAY_PHYSICS_DIVERGED: raw states and semantic metrics both mismatch")

    def _replay_verification_status(
        self, *, model_ref: str, model, data, payload: dict[str, Any]
    ) -> str:
        """重放 verification_status：审计 + 任务谓词复算（0915 §三/§十）。"""
        physical_audit_pass = self.audit(model_ref, trace_ref=payload["trace_ref"]).status == "PASS"
        predicates = payload.get("task_predicates")
        if not physical_audit_pass:
            return "FAIL"
        if predicates is None:
            return "NOT_EVALUATED"
        channels = sorted({p["channel"] for p in predicates})
        observations = observe_mod.observe_channels(model, data, channels)
        verdicts = evaluate_predicates(observations, predicates)
        return "PASS" if all(v["ok"] for v in verdicts) else "FAIL"

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

    def load_menagerie(self, model_name: str, *, entry: str | None = None) -> ModelReference:
        """Menagerie 正式接入（MH16，0916 §二十四）：官方
        mujoco_menagerie package（锁版本）导入模型到不可变 store。

        §24.2 绝不自动"最新版下载"——provenance 五元组全部记录：
        provider/package_version/model_revision/git oid/asset_digest/
        license。内容寻址幂等：同名同 ref。
        """
        try:
            import mujoco_menagerie as mm
        except ImportError as exc:
            raise ValueError(
                "MODEL_SOURCE_UNAVAILABLE: mujoco-menagerie 未安装（pyproject 已 pin）"
            ) from exc

        try:
            robot = mm.get(model_name)
        except Exception as exc:  # UnknownRobotError 及其族
            raise ValueError(f"MODEL_NOT_FOUND: menagerie 无此模型 {model_name!r}") from exc

        entry_name = entry or robot.default_model
        xml_path = robot.path() / f"{entry_name}.xml"
        if not xml_path.is_file():
            raise ValueError(
                f"MODEL_NOT_FOUND: {model_name!r} 无 entry point {entry_name!r}"
                f"（可用 {[e.name for e in robot.entry_points]}）"
            )
        ref = self.load_model_xml(
            xml_path.read_text(encoding="utf-8"),
            source={
                "kind": "menagerie",
                "provider": "menagerie",
                "package_version": str(mm.__version__),
                "model_name": model_name,
                "model_revision": str(robot.oid),
                "asset_digest": str(robot.sha256),
                "license": str(robot.license),
                "entry_point": entry_name,
            },
            assets=self._assets_from_file(xml_path),
        )
        return ref.model_copy(
            update={"runtime_capabilities": self.capabilities().to_canonical_dict()}
        )

    def scaffold_eurdf_from_menagerie(self, model_name: str, out_dir: Path) -> Path:
        """§24.3：从 menagerie 模型生成 e-URDF 声明脚手架——
        能力一律 UNDECLARED 起步（声明→证明绑定走既有 MH9 机制，
        Menagerie 模型不自动获得能力语义）。"""
        import shutil

        try:
            import mujoco_menagerie as mm
        except ImportError as exc:
            raise ValueError("MODEL_SOURCE_UNAVAILABLE: mujoco-menagerie 未安装") from exc
        try:
            robot = mm.get(model_name)
        except Exception as exc:
            raise ValueError(f"MODEL_NOT_FOUND: menagerie 无此模型 {model_name!r}") from exc

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(robot.path(), out_dir, dirs_exist_ok=True)
        (out_dir / "capabilities.yaml").write_text(
            "# e-URDF scaffold（MH16 §24.3）：从 menagerie 导入。\n"
            "# 所有能力默认 UNDECLARED——能力语义必须由 ROSClaw 显式声明，\n"
            "# 并经 MuJoCo 模型证明（声明→证明绑定，MH9 机制）。\n"
            f"# 来源: menagerie {model_name} @ {robot.oid}（license: {robot.license}）\n"
            "capabilities:\n"
            "  # 示例（取消注释并按模型真相填写后才算声明）：\n"
            "  # - name: grasp\n"
            "  #   required_hardware: [gripper]\n"
            "  #   status: UNDECLARED\n"
            "  - name: grasp\n"
            "    status: UNDECLARED\n",
            encoding="utf-8",
        )
        (out_dir / "semantic.yaml").write_text(
            "# 语义骨架（MH16 §24.3）：affordance 链接 UNDECLARED。\n"
            f"# 来源: menagerie {model_name} @ {robot.oid}\n"
            "links: {}\n"
            "affordances: {}\n",
            encoding="utf-8",
        )
        return out_dir

    def acceleration_compatibility(self, model_ref: str) -> dict[str, Any]:
        """Backend Fidelity Gate（MH18 §二十九）：静态兼容性分级
        CPU_ONLY / MJX_JAX_COMPATIBLE / MJX_WARP_COMPATIBLE + 原因。
        GPU 执行面：本机无 jax/warp 时诚实 NOT_RUN（分级是静态
        分析不依赖 GPU）。"""
        from rosclaw.sim import acceleration

        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)
        result = acceleration.compatibility(model, manifest["mjcf_xml"])
        result["model_ref"] = model_ref
        result["gpu_execution"] = "NOT_RUN"  # 本机无 jax/warp（诚实留痕）
        return result

    def shadow_compare(
        self, model_ref: str, observation_trace_ref: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Digital Shadow 比对（MH19 §三十三）：SIM 预测 vs REAL
        观测 → MATCH/DIVERGED + SysID 校准建议。"""
        from rosclaw.sim import shadow

        return shadow.shadow_compare(self, model_ref, observation_trace_ref, **kwargs)

    def import_observation(
        self,
        trace_ref: str,
        *,
        evidence_domain: str,
        body_id: str,
        body_snapshot_hash: str,
        source: dict[str, Any],
        joint_schema: list[dict[str, Any]],
        clock: dict[str, Any],
        calibration_ref: str,
        joint_order_in_trace: list[str] | None = None,
        model_ref: str | None = None,
    ) -> str:
        """导入观测为 ObservationTraceV2（MH21 §14-§16）：provenance
        全字段 + joint schema 按名校验（不按数组位置猜）+ trace
        按名重排为模型规范序（joint_order_in_trace ≠ 模型序时）。

        body_snapshot_hash 空 → 从观测源模型结构签名自动计算
        （REAL 日志桥接时由记录侧给出并在此复核）。
        """
        from rosclaw.sim import shadow as shadow_mod
        from rosclaw.sim.contracts import ObservationTraceV2

        trace = self.store.get(trace_ref)
        if not isinstance(trace, dict) or "states" not in trace:
            raise ValueError(f"SHADOW_OBSERVATION_EMPTY: {trace_ref} 不是 trace")
        source_model_ref = model_ref or trace.get("model_ref")
        if not source_model_ref:
            raise ValueError("OBSERVATION_INVALID: 缺 model_ref（无法校验 joint schema）")
        manifest = self._manifest(source_model_ref)
        spec = self._spec_from_manifest(manifest)
        model, _ = self._compile_smoke(spec)

        canonical_schema = shadow_mod.canonical_joint_schema(model)
        model_joint_names = {entry["joint"] for entry in canonical_schema}
        declared = {entry["joint"] for entry in joint_schema}
        if not declared or not declared <= model_joint_names:
            raise ValueError(
                "SHADOW_JOINT_SCHEMA_MISMATCH: joint_schema 与模型不符"
                f"（多余: {sorted(declared - model_joint_names)}）"
            )

        final_trace_ref = trace_ref
        if joint_order_in_trace is not None and joint_order_in_trace != [
            entry["joint"] for entry in canonical_schema if entry["joint"] in joint_order_in_trace
        ]:
            # 按名重排 trace 的 qpos/qvel 列（REAL joint state 顺序
            # 常与模型不同——按名映射，不按数组位置猜）。
            order_map = []
            for name in joint_order_in_trace:
                if name not in model_joint_names:
                    raise ValueError(f"SHADOW_JOINT_SCHEMA_MISMATCH: trace 列 {name!r} 不在模型中")
                order_map.append(
                    next(e for e in canonical_schema if e["joint"] == name)
                )
            states = trace["states"]
            reordered = []
            for row in states:
                qpos = row["qpos"]
                qvel = row["qvel"]
                new_qpos: list[float] = [0.0] * len(qpos)
                new_qvel: list[float] = [0.0] * len(qvel)
                src_qpos = 0
                src_qvel = 0
                for entry in order_map:
                    dst_qpos = int(entry["qpos_adr"])
                    dst_qvel = int(entry["dof_adr"])
                    # 宽度推断：下一地址差（hinge/slide=1, free=7/6）。
                    nq = len(qpos)
                    nv = len(qvel)
                    next_qpos = min(
                        (e["qpos_adr"] for e in canonical_schema if e["qpos_adr"] > dst_qpos),
                        default=nq,
                    )
                    next_dof = min(
                        (e["dof_adr"] for e in canonical_schema if e["dof_adr"] > dst_qvel),
                        default=nv,
                    )
                    w_qpos = next_qpos - dst_qpos
                    w_dof = next_dof - dst_qvel
                    new_qpos[dst_qpos : dst_qpos + w_qpos] = qpos[src_qpos : src_qpos + w_qpos]
                    new_qvel[dst_qvel : dst_qvel + w_dof] = qvel[src_qvel : src_qvel + w_dof]
                    src_qpos += w_qpos
                    src_qvel += w_dof
                reordered.append({**row, "qpos": new_qpos, "qvel": new_qvel})
            new_trace = {**trace, "states": reordered}
            final_trace_ref = self.store.put("traces", new_trace)

        if not body_snapshot_hash:
            body_snapshot_hash = shadow_mod.body_snapshot_hash(model)
        observation = ObservationTraceV2(
            backend="mujoco",
            evidence_domain=evidence_domain,
            body_id=body_id,
            body_snapshot_hash=body_snapshot_hash,
            source=source,
            joint_schema=canonical_schema,
            clock=clock,
            calibration_ref=calibration_ref,
            channels=["qpos", "qvel"],
            trace_ref=final_trace_ref,
        )
        return self.store.put("traces", observation.to_canonical_dict() | {"kind": "observation_trace_v2"})

    def record_dataset(self, model_ref: str, *, sequences: list[dict[str, Any]]) -> str:
        """录制 SysID 数据集（MH17）：每序列 = 初始状态 + 受控 rollout
        trace。内容寻址幂等（同参数重录同 ref）。

        合成层从 truth 模型录制；真实层由 REAL 日志桥接为同一形态
        （Digital Twin 管线 §二十六）。
        """
        import mujoco

        manifest = self._manifest(model_ref)
        spec = self._spec_from_manifest(manifest)
        seq_records = []
        for seq in sequences:
            controller = seq.get("controller") or {"hold": True}
            duration_s = seq.get("duration_s")
            if duration_s is None or float(duration_s) <= 0:
                raise ValueError("SYSID_DATASET_INVALID: duration_s 必须为正")
            qpos0 = seq.get("qpos0")
            model = spec.compile()
            data = mujoco.MjData(model)
            if qpos0 is not None:
                if len(qpos0) != model.nq:
                    raise ValueError(
                        f"SYSID_DATASET_INVALID: qpos0 维度 {len(qpos0)} != nq {model.nq}"
                    )
                data.qpos[:] = [float(v) for v in qpos0]
            mujoco.mj_forward(model, data)
            state_ref = self.capture_and_store_v2(model_ref, model, data)
            receipt = self.run_experiment(
                model_ref,
                state_ref=state_ref,
                controller=controller,
                duration_s=float(duration_s),
                audit=False,
            )
            seq_records.append(
                {
                    "trace_ref": receipt.trace_ref,
                    "initial_state_ref": receipt.initial_state_ref,
                    "controller": controller,
                    "duration_s": float(duration_s),
                    "qpos0": [float(v) for v in qpos0] if qpos0 is not None else None,
                }
            )
        dataset = {
            "kind": "sysid_dataset",
            "schema_version": "rosclaw.sim.dataset.v1",
            "model_ref": model_ref,
            "sequences": seq_records,
        }
        return self.store.put("traces", dataset)

    def run_sysid(self, spec: dict[str, Any]) -> dict[str, Any]:
        """System Identification（MH17）：算法核心复用官方
        mujoco.sysid 工具箱；候选模型经 patch 血缘派生；holdout
        独立复算后才允许 IMPROVED（§26.3）。"""
        from rosclaw.sim import sysid

        return sysid.run_sysid(self, spec)

    def _load_assets(self, manifest: dict[str, Any]) -> dict[str, bytes]:
        assets: dict[str, bytes] = {}
        for name, ref in manifest["assets"].items():
            blob = self.store.get(ref)
            if not isinstance(blob, bytes):
                raise ValueError(f"STORE_DIGEST_MISMATCH: asset {name!r} is not raw bytes")
            assets[name] = blob
        return assets

    def _spec_from_manifest(self, manifest: dict[str, Any]):  # noqa: ANN202
        return _spec_from_xml_assets(manifest["mjcf_xml"], self._load_assets(manifest))

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

    def _model_digest(self, manifest: dict[str, Any]) -> str:
        """真正的物理模型身份（PR-MH9，0915 文档 §四）：
        canonical MJCF + 全部资产 blob digest（排序确定性）。

        相同 XML、不同 mesh 字节 → 不同 model_digest → state/trace/
        replay 绑定全部 fail closed（CROSS_MODEL_REF）。
        """
        payload = {
            "xml": hashlib.sha256(manifest["mjcf_xml"].encode("utf-8")).hexdigest(),
            "assets": sorted(
                [name, hashlib.sha256(blob).hexdigest()]
                for name, blob in self._load_assets(manifest).items()
            ),
        }
        return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

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
    rounded: dict[str, Any] = {}
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


_RENDER_WORKER_CODE = r"""
import json
import os
import sys

backend = sys.argv[1]
os.environ["MUJOCO_GL"] = backend

import mujoco
from PIL import Image

request = json.loads(open(sys.argv[2], encoding="utf-8").read())
model = mujoco.MjModel.from_xml_path(request["model_path"])
renderer = mujoco.Renderer(model, request["height"], request["width"])
frames = []
for snapshot in request["states"]:
    data = mujoco.MjData(model)
    data.time = float(snapshot["t"])
    for i, v in enumerate(snapshot["qpos"]):
        data.qpos[i] = v
    for i, v in enumerate(snapshot["qvel"]):
        data.qvel[i] = v
    for i, v in enumerate(snapshot["ctrl"]):
        data.ctrl[i] = v
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=request.get("camera") or -1)
    frames.append(Image.fromarray(renderer.render()))
renderer.close()
frames[0].save(
    request["out"], format="GIF", save_all=True,
    append_images=frames[1:], duration=80, loop=0,
)
"""


def _spec_from_xml_assets(xml_text: str, assets: dict[str, bytes]):  # noqa: ANN202
    """MjSpec 从 XML+资产构建（MH10b 实证结论）。

    3.13.0 绑定实证：`MjSpec.from_file/from_string(vfs=)` 对 meshdir
    资产**不解析 VFS**（Error opening file）；VFS 仅在
    `MjModel.from_xml_path(name, vfs=)` 完整工作，`.mjz` 便携工件经
    `spec.assets` 填充 + `to_zip/from_zip` 往返自包含。因此 MjSpec
    资产面继续使用 `assets=`（3.13 实测零弃用警告），MjVfs 绑定
    支持后随版本迁移（ADR-0014 MH10b 段）。
    """
    import mujoco

    try:
        return mujoco.MjSpec.from_string(xml_text, assets=assets or None)
    except Exception as exc:
        raise ValueError(f"MODEL_COMPILE_FAILED: {exc}") from exc


_CAMERA_WORKER_CODE = r"""
import json
import os
import sys

backend = sys.argv[1]
os.environ["MUJOCO_GL"] = backend

import mujoco
import numpy as np
from PIL import Image

request = json.loads(open(sys.argv[2], encoding="utf-8").read())
model = mujoco.MjModel.from_xml_path(request["model_path"])
data = mujoco.MjData(model)
state = request["state"]
data.time = float(state["time"])
for i, v in enumerate(state["qpos"]):
    data.qpos[i] = v
for i, v in enumerate(state["qvel"]):
    data.qvel[i] = v
for i, v in enumerate(state["ctrl"]):
    data.ctrl[i] = v
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, request["height"], request["width"])
renderer.update_scene(data, camera=request["camera"])
kind = request["kind"]
if kind == "camera_depth":
    renderer.enable_depth_rendering()
    depth = np.asarray(renderer.render(), dtype=np.float64)
    finite = depth[np.isfinite(depth)]
    norm = np.zeros_like(depth)
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        norm = np.clip((depth - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    Image.fromarray((norm * 65535).astype(np.uint16), mode="I;16").save(request["out"])
elif kind == "camera_segmentation":
    renderer.enable_segmentation_rendering()
    seg = np.asarray(renderer.render())[:, :, 0].astype(np.uint8)
    Image.fromarray(seg).save(request["out"])
else:
    Image.fromarray(renderer.render()).save(request["out"])
renderer.close()
"""
