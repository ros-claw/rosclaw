"""MujocoBackend（PR-MH2，ADR-0014，规格 §12）。

Maturity: experimental（ADR-0000 §4）。

模型服务四方法：load_model / inspect_model / compile_model / patch_model。

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

from rosclaw.contracts.common import content_hash
from rosclaw.sim.backends.mujoco.inspect import inspect_model_full
from rosclaw.sim.backends.mujoco.patch import apply_patches
from rosclaw.sim.capabilities import probe_mujoco_capabilities
from rosclaw.sim.contracts import (
    ModelInspection,
    ModelPatchResult,
    ModelReference,
    SimulationBackendCapabilities,
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
