"""0902 审计 R2-1 红测试：RenderSpec/RenderProfile/ToolAsset 契约
（§4.3 通用渲染器 + R2.1 契约完成）。

0902 假成功根因之一：渲染是五角星专用画线逻辑，需求里的"红色圆柱
笔/3D 场景/实际轨迹 overlay"没有契约位置——链复用旧视频冒充新
需求。§4.3：RenderSpec 是一等契约（body_ref/world_ref/attachments/
overlays/cameras/outputs），任何本体+附件+世界+overlay+相机组合
都走同一渲染路径。

闭环断言：
1. 三契约注册进 ALL_CONTRACTS + golden schema 稳定；
2. RenderSpec overlay kind 是冻结枚举（actual_eef_trace/
   planned_trace/waypoints/contact_points/safety_zone/sensor_frustum）
   ——未知 kind 拒绝（不静默吞）；
3. camera preset 冻结枚举（follow/free/top）；outputs 非空且属于
   mp4/gif；
4. ToolAsset 必须声明 physical——visual-only 附件不得参与接触
   声明（§4.3：可视化附件不得冒充真实接触）；
5. RenderProfile 必须有 eef_frame 与 qpos 映射（渲染任意本体靠它，
   不许本体名 hardcode）；
6. RenderSpec 校验 attachments 的 mount_frame 非空、overlays 的
   source_ref 可解析形态（trace:.../plan:...）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from rosclaw.contracts.export import ALL_CONTRACTS


class TestRenderContractsRegistered:
    def test_contracts_in_registry(self) -> None:
        assert "rosclaw.render_spec.v1" in ALL_CONTRACTS
        assert "rosclaw.render_profile.v1" in ALL_CONTRACTS
        assert "rosclaw.tool_asset.v1" in ALL_CONTRACTS

    def test_golden_files_exist(self) -> None:
        from pathlib import Path

        golden = Path(__file__).parent / "contracts" / "golden"
        for stem in ("render_spec.v1", "render_profile.v1", "tool_asset.v1"):
            assert (golden / f"rosclaw.{stem}.json").exists(), f"缺 golden: {stem}"


class TestRenderSpec:
    def _base(self) -> dict:
        return {
            "body_ref": "robot:sim/ur5e",
            "world_ref": "world:tabletop",
            "outputs": ["mp4"],
        }

    def test_minimal_valid(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderSpecV1

        spec = RenderSpecV1(**self._base())
        assert spec.body_ref == "robot:sim/ur5e"

    def test_overlay_kind_frozen_enum(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderSpecV1

        base = self._base()
        base["overlays"] = [{"kind": "actual_eef_trace", "source_ref": "trace:t1"}]
        RenderSpecV1(**base)  # 合法 kind 通过
        base["overlays"] = [{"kind": "magic_overlay", "source_ref": "trace:t1"}]
        with pytest.raises(PydanticValidationError):
            RenderSpecV1(**base)

    def test_overlay_source_ref_required_for_trace_kinds(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderSpecV1

        base = self._base()
        base["overlays"] = [{"kind": "actual_eef_trace"}]
        with pytest.raises(PydanticValidationError):
            RenderSpecV1(**base)

    def test_camera_preset_frozen_enum(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderSpecV1

        base = self._base()
        base["cameras"] = [{"preset": "follow"}]
        RenderSpecV1(**base)
        base["cameras"] = [{"preset": "hollywood"}]
        with pytest.raises(PydanticValidationError):
            RenderSpecV1(**base)

    def test_outputs_frozen_and_nonempty(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderSpecV1

        base = self._base()
        base["outputs"] = []
        with pytest.raises(PydanticValidationError):
            RenderSpecV1(**base)
        base["outputs"] = ["avi"]
        with pytest.raises(PydanticValidationError):
            RenderSpecV1(**base)

    def test_attachment_requires_mount_frame(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderSpecV1

        base = self._base()
        base["attachments"] = [{"tool_ref": "tool:gripper", "mount_frame": ""}]
        with pytest.raises(PydanticValidationError):
            RenderSpecV1(**base)


class TestToolAsset:
    def test_physical_flag_required_and_honest(self) -> None:
        from rosclaw.contracts.agent.tool_asset import ToolAssetV1

        # visual-only 附件合法登记，但 physical=False 必须显式。
        asset = ToolAssetV1(
            tool_id="tool:marker_red",
            name="红色标记笔",
            physical=False,
            adapters={"mjcf": "tools/marker_red.xml"},
            mount={"parent_frame": "ee_link"},
        )
        assert asset.physical is False
        with pytest.raises(PydanticValidationError):
            # 缺 physical 字段——默认 True 会让可视化附件冒充实体。
            ToolAssetV1(
                tool_id="tool:x", name="x",
                adapters={"mjcf": "x.xml"}, mount={"parent_frame": "ee"},
            )

    def test_adapter_format_frozen(self) -> None:
        from rosclaw.contracts.agent.tool_asset import ToolAssetV1

        with pytest.raises(PydanticValidationError):
            ToolAssetV1(
                tool_id="tool:x", name="x", physical=True,
                adapters={"blend": "x.blend"},  # 非法适配格式
                mount={"parent_frame": "ee"},
            )


class TestRenderProfile:
    def test_requires_eef_and_qpos_mapping(self) -> None:
        from rosclaw.contracts.agent.render_spec import RenderProfileV1

        profile = RenderProfileV1(
            body_id="sim/ur5e",
            root_body="base_link",
            eef_frame="ee_link",
            qpos_mapping={"shoulder_pan_joint": 0},
        )
        assert profile.eef_frame == "ee_link"
        with pytest.raises(PydanticValidationError):
            RenderProfileV1(body_id="sim/ur5e", root_body="base_link")  # 缺 eef/qpos


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
