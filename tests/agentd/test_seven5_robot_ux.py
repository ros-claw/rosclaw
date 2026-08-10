"""PR-SEVEN-5 红测试（七审 §6）：Robot-first UX 与自修复。

红测试先行——当前缺陷：

1. pi.status 不带机器人友好名/kit 状态——UI 只能显示内部 body_id；
2. 没有 pi.robot.list / pi.robot.resolve——自然语言"机械臂"无法
   解析到可用 kit；
3. 没有 pi.doctor.task——"画五角星"需要 trajectory+executor+verifier
   的就绪检查无处可问；
4. kit BROKEN 时没有结构化 remediation（模型只会说"请重新绑定
   profile"）；没有幂等的 pi.robot.repair / pi.robot.use。
"""

from __future__ import annotations

from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _turn


async def _service(tmp_path: Path, *, disable_kit: bool = False):
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.service import AgentService

    config_path = tmp_path / "config.yaml"
    if disable_kit:
        config_path.write_text(
            "kits:\n  disabled: [rosclaw/ur5e-sim]\n", encoding="utf-8"
        )
    config = load_agent_config(config_path)
    return AgentService(
        config, tmp_path, gateway=MockModelGateway(mock_profile(), [_turn()] * 4)
    )


def _bridge(service, tmp_path: Path):
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    return PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")


class TestRobotStatusAndList:
    async def test_status_carries_body_display_and_kit(self, tmp_path: Path) -> None:
        """pi.status 带友好名 + kit 摘要——Header 不再只显示内部 body_id。"""
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.status", {"token": service.control_token}
        )
        assert result.get("ok")
        assert result.get("body_display") == "UR5e（本地仿真）", result
        kit = result.get("robot_kit") or {}
        assert kit.get("kit_id") == "rosclaw/ur5e-sim"
        assert kit.get("state") == "READY"
        await service.close()

    async def test_robot_list_marks_active(self, tmp_path: Path) -> None:
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.robot.list", {"token": service.control_token}
        )
        assert result.get("ok"), result
        kits = result.get("kits") or []
        ur5e = next((k for k in kits if k.get("kit_id") == "rosclaw/ur5e-sim"), None)
        assert ur5e is not None, f"第一方 UR5e kit 未列出: {kits}"
        assert ur5e.get("active") is True
        assert ur5e.get("state") == "READY"
        assert ur5e.get("display_name") == "UR5e（本地仿真）"
        await service.close()


class TestRobotResolve:
    async def test_arm_intent_selects_ur5e(self, tmp_path: Path) -> None:
        """自然语言"机械臂"→ 唯一候选自动选 UR5e kit。"""
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.robot.resolve",
            {"token": service.control_token, "query": "我想让机械臂画五角星"},
        )
        assert result.get("ok"), result
        assert result.get("selected", {}).get("kit_id") == "rosclaw/ur5e-sim"
        await service.close()

    async def test_unknown_robot_honest_empty(self, tmp_path: Path) -> None:
        """无匹配 kit（limo 小车）→ 诚实空候选，不伪造匹配。"""
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.robot.resolve",
            {"token": service.control_token, "query": "limo 小车巡航"},
        )
        assert result.get("ok"), result
        assert result.get("selected") is None
        assert result.get("candidates") == []
        await service.close()


class TestDoctorTask:
    async def test_star_task_ready(self, tmp_path: Path) -> None:
        """画五角星需要 trajectory+executor+verifier——默认安装全 READY。"""
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.doctor.task",
            {"token": service.control_token, "goal": "画五角星"},
        )
        assert result.get("ok"), result
        assert result.get("state") == "READY", result
        required = set(result.get("required") or [])
        assert {"trajectory", "executor", "verifier"} <= required
        await service.close()

    async def test_star_task_missing_structured_remediation(
        self, tmp_path: Path
    ) -> None:
        """kit 被禁用 → MISSING + 结构化 remediation（幂等/可取消/
        绝不自动完成 REAL 授权）——不再是"请重新绑定 profile"。"""
        service = await _service(tmp_path, disable_kit=True)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.doctor.task",
            {"token": service.control_token, "goal": "draw a five-pointed star"},
        )
        assert result.get("ok"), result
        assert result.get("state") == "MISSING", result
        missing = set(result.get("missing") or [])
        assert {"trajectory", "executor", "verifier"} <= missing
        remediation = result.get("remediation") or {}
        assert remediation.get("kind") == "enable_robot_kit"
        assert remediation.get("kit_id") == "rosclaw/ur5e-sim"
        assert remediation.get("idempotent") is True
        assert remediation.get("cancellable") is True
        assert remediation.get("real_authorization") is False
        await service.close()


class TestDoctorTaskCli:
    def test_cli_doctor_task_ready(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """rosclaw doctor task（静态 manifest 检查）：默认安装画五角星
        READY；kit 禁用 → MISSING + remediation。"""
        from rosclaw.cli import _run_doctor_task

        monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("", encoding="utf-8")
        assert _run_doctor_task("画五角星", json_output=True) == 0
        import json

        report = json.loads(capsys.readouterr().out)
        assert report["state"] == "READY"
        assert {"trajectory", "executor", "verifier"} <= set(report["required"])

        (tmp_path / "config.yaml").write_text(
            "kits:\n  disabled: [rosclaw/ur5e-sim]\n", encoding="utf-8"
        )
        assert _run_doctor_task("画五角星", json_output=True) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["state"] == "MISSING"
        assert report["remediation"]["kind"] == "enable_robot_kit"
        assert report["remediation"]["real_authorization"] is False


class TestRepairAndUse:
    async def test_repair_reenables_disabled_kit_idempotent(
        self, tmp_path: Path
    ) -> None:
        """repair 把被禁用的 kit 重新启用——幂等（两次调用都 ok）。"""
        service = await _service(tmp_path, disable_kit=True)
        bridge = _bridge(service, tmp_path)
        first = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.robot.repair",
            {"token": service.control_token, "kit_id": "rosclaw/ur5e-sim"},
        )
        assert first.get("ok"), first
        assert (first.get("robot_kit") or {}).get("state") == "READY", first
        second = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.robot.repair",
            {"token": service.control_token, "kit_id": "rosclaw/ur5e-sim"},
        )
        assert second.get("ok"), second
        assert (second.get("robot_kit") or {}).get("state") == "READY"
        await service.close()

    async def test_use_same_body_idempotent(self, tmp_path: Path) -> None:
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.robot.use",
            {"token": service.control_token, "body_id": "sim/ur5e"},
        )
        assert result.get("ok"), result
        assert result.get("changed") is False
        assert result.get("body_id") == "sim/ur5e"
        await service.close()

    async def test_use_unknown_body_refused(self, tmp_path: Path) -> None:
        """无 kit 的 body（含 REAL 真机）一律拒绝——robot use 绝不
        自动完成真机授权。"""
        service = await _service(tmp_path)
        bridge = _bridge(service, tmp_path)
        result = await bridge._dispatch(
            "user:local:1000",
            1,
            "pi.robot.use",
            {"token": service.control_token, "body_id": "real/xarm-7"},
        )
        assert not result.get("ok"), "无 kit 的 REAL body 竟可 use"
        assert result.get("code") in ("BODY_UNKNOWN", "MODE_FORBIDDEN", "PROFILE_FORBIDDEN")
        await service.close()
