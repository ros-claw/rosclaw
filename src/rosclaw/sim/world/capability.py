"""能力声明→证明绑定（PR-MH9，0915 优化文档 §五）。

语义由 Body Profile（e-URDF capabilities/semantic 或 task sidecar）
**声明**，物理模型负责**证明**它没有说谎——不再从 actuator 名字
反推语义（`"gripper" in name` 是改名版的猜名字）。

三态（fail closed）：
- AVAILABLE：声明了，且模型证明对应执行器真实存在；
- UNDECLARED：没有声明通道或声明中无 grasp 能力；
- UNPROVEN：声明了 grasp，但模型里没有可验证的执行器。
UNDECLARED / UNPROVEN 一律拒绝 grasp（CAPABILITY_UNAVAILABLE）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

GRASP_AVAILABLE = "AVAILABLE"
GRASP_UNDECLARED = "UNDECLARED"
GRASP_UNPROVEN = "UNPROVEN"


def _eurdf_declares_grasp(zoo_dir: Path) -> bool:
    caps_file = zoo_dir / "capabilities.yaml"
    if not caps_file.is_file():
        return False
    data = yaml.safe_load(caps_file.read_text(encoding="utf-8")) or {}
    for capability in data.get("capabilities", []) or []:
        hardware = capability.get("required_hardware", []) or []
        constraints = capability.get("constraints", {}) or {}
        if "gripper" in hardware or constraints.get("requires_gripper") is True:
            return True
    return False


def _eurdf_gripper_links(zoo_dir: Path) -> set[str]:
    """semantic.yaml 中 affordances 含 gripper/grasp 的 region link——
    声明给出的"夹爪在哪条链上"的绑定通道。"""
    semantic_file = zoo_dir / "semantic.yaml"
    if not semantic_file.is_file():
        return set()
    data = yaml.safe_load(semantic_file.read_text(encoding="utf-8")) or {}
    links: set[str] = set()
    for region in data.get("functional_regions", []) or []:
        affordances = region.get("affordances", []) or []
        if any("gripper" in str(a) or "grasp" in str(a) for a in affordances):
            link = region.get("link")
            if link:
                links.add(str(link))
    return links


def _subtree_has_actuated_joint(model, root_body_id: int) -> bool:  # noqa: ANN001
    import mujoco  # noqa: F401

    subtree = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if int(model.body_parentid[body_id]) in subtree and body_id not in subtree:
                subtree.add(body_id)
                changed = True
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id >= 0 and int(model.jnt_bodyid[joint_id]) in subtree:
            return True
    return False


def _prove_with_links(backend, model_ref: str, links: set[str]) -> bool:  # noqa: ANN001
    if not links:
        return False
    import mujoco

    manifest = backend._manifest(model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    for link in links:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        if body_id >= 0 and _subtree_has_actuated_joint(model, body_id):
            return True
    return False


def _prove_task_sidecar(backend, body_ref: dict) -> dict[str, Any]:  # noqa: ANN001
    """task sidecar <model>.capabilities.yaml：
    grasp.actuator_joints 声明的 joint 必须存在且有 actuator 驱动。"""
    import mujoco

    ref_path = Path(body_ref["ref"])
    sidecar = backend._task_root / ref_path.with_suffix(".capabilities.yaml")
    if not sidecar.is_file():
        return {"status": GRASP_UNDECLARED, "detail": "no task capabilities sidecar"}
    data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    grasp = data.get("grasp", {}) or {}
    declared_joints = grasp.get("actuator_joints", []) or []
    if not declared_joints:
        return {"status": GRASP_UNDECLARED, "detail": "sidecar declares no grasp actuator_joints"}

    manifest = backend._manifest(body_ref["_model_ref"])
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    actuated_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i][0]))
        for i in range(model.nu)
        if int(model.actuator_trnid[i][0]) >= 0
    }
    missing = [name for name in declared_joints if name not in actuated_joints]
    if missing:
        return {
            "status": GRASP_UNPROVEN,
            "detail": f"declared joints not actuated in model: {missing}",
        }
    return {"status": GRASP_AVAILABLE, "detail": f"proven joints: {sorted(declared_joints)}"}


def resolve_grasp_capability(backend, body_ref: dict, *, model_ref: str) -> dict[str, Any]:  # noqa: ANN001
    """解析 body_ref 的 grasp 能力三态。

    body_ref: WorldSpec 的 {"id", "kind", "ref"}；model_ref: 已加载模型。
    """
    kind = body_ref["kind"]
    if kind == "task":
        return _prove_task_sidecar(backend, {**body_ref, "_model_ref": model_ref})

    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    zoo_dir = _default_zoo_path() / body_ref["ref"]
    if not _eurdf_declares_grasp(zoo_dir):
        return {"status": GRASP_UNDECLARED, "detail": "e-URDF declares no grasp capability"}
    links = _eurdf_gripper_links(zoo_dir)
    if _prove_with_links(backend, model_ref, links):
        return {"status": GRASP_AVAILABLE, "detail": f"proven via links: {sorted(links)}"}
    return {
        "status": GRASP_UNPROVEN,
        "detail": "e-URDF declares grasp, but model has no actuated gripper subtree"
        f" (links tried: {sorted(links) or 'none'})",
    }
