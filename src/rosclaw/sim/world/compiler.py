"""WorldSpec → MJCF 编译器（PR-MH7，规格 §22/§42）。

WorldSpec 是语义，MJCF 是物理实现：ground/objects/markers/cameras
由 spec 直接构造；body_refs 的机器人经 backend 加载后以
``MjSpec.attach`` 挂入世界（prefix 命名空间隔离），资产按 basename
复用内容寻址 blob（不重复存储）。

能力诚实（§24）：grasp 类 affordance 要求被挂 body 实际有夹爪
执行器（编译后推导，不从名字猜），否则 CAPABILITY_UNAVAILABLE。
"""

from __future__ import annotations

from typing import Any

from rosclaw.sim.world.capability import resolve_grasp_capability
from rosclaw.sim.world.interactions import check_body_capability, interaction_order
from rosclaw.sim.world.validation import validate_worldspec

_SHAPE_TYPES = {
    "box": "mjGEOM_BOX",
    "cylinder": "mjGEOM_CYLINDER",
    "sphere": "mjGEOM_SPHERE",
    "capsule": "mjGEOM_CAPSULE",
}


def compile_world(backend, worldspec: dict[str, Any], *, name: str = "world") -> dict[str, Any]:  # noqa: ANN001
    """校验并编译 WorldSpec，返回世界模型 ref 与交互清单。"""
    normalized = validate_worldspec(worldspec)

    import mujoco

    spec = mujoco.MjSpec()
    spec.option.timestep = normalized["world"]["timestep_s"]
    spec.option.gravity[:] = normalized["world"]["gravity"]
    if normalized["world"]["ground"]:
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5.0, 5.0, 0.1],
        )

    for camera in normalized["sensors"]:
        spec.worldbody.add_camera(name=camera["name"], pos=camera["pos"], quat=camera["quat"])

    for obj in normalized["objects"]:
        body = spec.worldbody.add_body(name=obj["id"], pos=obj["pos"], quat=obj["quat"])
        if obj["dynamic"]:
            body.add_joint(name=f"{obj['id']}_free", type=mujoco.mjtJoint.mjJNT_FREE)
        body.add_geom(
            name=f"{obj['id']}_geom",
            type=getattr(mujoco.mjtGeom, _SHAPE_TYPES[obj["shape"]]),
            size=obj["size"],
            mass=obj["mass"],
            rgba=obj["rgba"],
            friction=obj["friction"],
        )

    for point in normalized["interaction_points"]:
        parent = spec.worldbody
        marker_pos = [0.0, 0.0, 0.0]
        if point["target"]["type"] == "body":
            for obj in normalized["objects"]:
                if obj["id"] == point["target"]["name"]:
                    parent = next(b for b in spec.bodies if b.name == obj["id"])
                    # marker 贴物体顶面（原点会埋入 geom——A08 会抓）
                    marker_pos = [
                        0.0,
                        0.0,
                        obj["size"][2] if obj["shape"] != "sphere" else obj["size"][0],
                    ]
                    break
        parent.add_site(name=f"marker_{point['id']}", pos=marker_pos, group=2)

    assets: dict[str, bytes] = {}
    capabilities: dict[str, dict[str, Any]] = {}
    for body_ref in normalized["body_refs"]:
        robot_ref = backend.load_model(body_ref["ref"])
        robot_manifest = backend.store.get(robot_ref.model_ref)
        robot_assets = {
            key: backend.store.get(ref) for key, ref in robot_manifest["assets"].items()
        }
        robot_spec = mujoco.MjSpec.from_string(
            robot_manifest["mjcf_xml"], assets=robot_assets or None
        )
        frame = spec.worldbody.add_frame(pos=body_ref["pose"]["pos"], quat=body_ref["pose"]["quat"])
        spec.attach(robot_spec, frame=frame, prefix=f"{body_ref['id']}_")
        # attach 后 mesh 引用带 prefix（3.11 实测）；剥 prefix 按 basename
        # 复用内容寻址 blob。
        prefix = f"{body_ref['id']}_"
        for mesh in spec.meshes:
            filename = getattr(mesh, "file", "") or ""
            if not filename or filename in assets:
                continue
            candidate = filename[len(prefix) :] if filename.startswith(prefix) else filename
            for key, data in robot_assets.items():
                if key.rsplit("/", 1)[-1] == candidate:
                    assets[filename] = data
                    break
        # 能力声明→证明绑定（0915 §五）：不从 actuator 名字猜夹爪。
        capabilities[body_ref["id"]] = resolve_grasp_capability(
            backend, body_ref, model_ref=robot_ref.model_ref
        )

    # 无假 affordance（§24）：grasp 要求声明+证明都成立的夹爪。
    check_body_capability(normalized["interaction_points"], capabilities=capabilities)

    result = backend.load_model_xml(
        spec.to_xml(),
        source={"kind": "worldspec", "ref": name},
        assets=assets,
        extra_manifest={"worldspec": normalized},
    )
    return {
        "model_ref": result.model_ref,
        "model_digest": result.model_digest,
        "worldspec": normalized,
        "interaction_order": interaction_order(normalized["interaction_points"]),
        "interaction_points": normalized["interaction_points"],
        "capabilities": {key: verdict["status"] for key, verdict in capabilities.items()},
        "capabilities_detail": capabilities,
    }
