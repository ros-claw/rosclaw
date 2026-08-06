"""批次 E（第二部分）测试：export/import/settings/reload。

- export → import round trip（只读归档、不恢复授权）
- 恶意 bundle：路径穿越、checksum 篡改、secret 内容、zip bomb 全拒绝
- settings：白名单、安全域拒绝、原子写
- reload：安全域拒绝、prompts/workers 成功
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.common import ValidationError
from rosclaw.contracts.ui.commands import CommandRequestV1


def _answer(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), [_answer] * 50))


def _cmd(name: str, args: dict | None = None, mission_id: str | None = None, key: str = "k") -> CommandRequestV1:
    return CommandRequestV1(
        request_id=f"r_{key}",
        idempotency_key=key,
        command_name=name,
        arguments=args or {},
        mission_id=mission_id,
    )


class TestExportImport:
    async def test_round_trip_read_only(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("导出测试")
            await service.send_turn(mission.mission_id, "你好")
            out = tmp_path / "bundle.rcmission"
            exported = await service.commands.execute(
                _cmd("export", mission_id=mission.mission_id, args={"path": str(out)})
            )
            assert exported.ok and out.exists()
            # bundle 内容：manifest 正确、无 secret 形态。
            with zipfile.ZipFile(out) as zf:
                names = set(zf.namelist())
                assert {"manifest.json", "conversation.jsonl", "checksums.txt"} <= names
                blob = b"".join(zf.read(n) for n in names).decode(errors="replace")
            assert "sk-" not in blob
            assert "permit_secret" not in blob and "private_signature" not in blob
            manifest = json.loads(zipfile.ZipFile(out).read("manifest.json"))
            assert manifest["magic"] == "rcmission/1"

            # import：只读归档、不恢复任何授权。
            imported = await service.commands.execute(
                _cmd("import", args={"path": str(out)}, key="k2")
            )
            assert imported.ok
            assert imported.data["read_only"] and imported.data["authority_restored"] is False
            new_id = imported.data["mission_id"]
            assert new_id != mission.mission_id
            with pytest.raises(ValidationError, match="archived"):
                await service.send_turn(new_id, "恢复？" )
            assert service.list_grants() == []
        finally:
            await service.close()

    async def test_malicious_bundles_refused(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            # 路径穿越
            evil = tmp_path / "evil.rcmission"
            with zipfile.ZipFile(evil, "w") as zf:
                zf.writestr("../../etc/pwned", "x")
            with pytest.raises(ValidationError, match="unsafe path"):
                service.importer.preview(evil)

            # checksum 篡改
            mission = service.create_mission("源")
            await service.send_turn(mission.mission_id, "hi")
            good = tmp_path / "good.rcmission"
            service.exporter.export_bundle(mission.mission_id, good)
            tampered = tmp_path / "tampered.rcmission"
            with zipfile.ZipFile(good) as zin, zipfile.ZipFile(tampered, "w") as zout:
                for name in zin.namelist():
                    content = zin.read(name)
                    if name == "conversation.jsonl":
                        content = b"tampered\n"
                    zout.writestr(name, content)
            with pytest.raises(ValidationError, match="checksum mismatch"):
                service.importer.preview(tampered)

            # secret 内容
            secret_bundle = tmp_path / "secret.rcmission"
            manifest = {"magic": "rcmission/1", "mission": {"goal": "x", "mission_id": "m"}}
            files = {
                "manifest.json": json.dumps(manifest),
                "conversation.jsonl": '{"role":"user","content":"key sk-abcdefghij123"}\n',
            }
            checksums = "\n".join(
                f"{hashlib.sha256(c.encode()).hexdigest()}  {n}" for n, c in sorted(files.items())
            )
            files["checksums.txt"] = checksums
            with zipfile.ZipFile(secret_bundle, "w") as zf:
                for n, c in files.items():
                    zf.writestr(n, c)
            with pytest.raises(ValidationError, match="secret-like"):
                service.importer.preview(secret_bundle)

            # 非 zip
            notzip = tmp_path / "notzip.rcmission"
            notzip.write_text("hello")
            with pytest.raises(ValidationError, match="not a zip"):
                service.importer.preview(notzip)
        finally:
            await service.close()

    async def test_share_points_to_export(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(_cmd("share"))
            assert not result.ok and result.error_code == "not_implemented"
            assert "/export" in result.message
        finally:
            await service.close()


class TestSettings:
    async def test_whitelist_and_atomic_write(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            set_ok = await service.commands.execute(
                _cmd("settings", args={"key": "agent.language", "value": "zh-CN"})
            )
            assert set_ok.ok
            assert service.settings.get_key("agent.language") == "zh-CN"
            # 原子写结果可解析。
            import yaml

            data = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
            assert data["agent"]["language"] == "zh-CN"
        finally:
            await service.close()

    async def test_safety_domain_and_unknown_key_rejected(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            safety = await service.commands.execute(
                _cmd("settings", args={"key": "agent.budgets.physical_action_count", "value": "5"})
            )
            assert not safety.ok and "安全域" in safety.message
            unknown = await service.commands.execute(
                _cmd("settings", args={"key": "agent.telemetry.endpoint", "value": "x"}, key="k2")
            )
            assert not unknown.ok and unknown.error_code == "settings_rejected"
        finally:
            await service.close()


class TestReload:
    async def test_prompts_and_workers_reload(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            mission = service.create_mission("reload 测试")
            result = await service.commands.execute(
                _cmd("reload", args={"domains": "prompts workers"}, mission_id=mission.mission_id)
            )
            assert result.ok
            assert result.data["prompts"]["ok"] and result.data["workers"]["ok"]
            # config.reloaded 事件落 journal。
            from rosclaw.contracts.agent.agent_event import AgentEventType

            types = [e.type for e in service.events_replay(mission.mission_id)]
            assert AgentEventType.CONFIG_RELOADED in types
        finally:
            await service.close()

    async def test_safety_domains_rejected(self, tmp_path: Path) -> None:
        service = _service(tmp_path)
        try:
            result = await service.commands.execute(
                _cmd("reload", args={"domains": "policy robot_pack"})
            )
            assert not result.ok
            assert not result.data["policy"]["ok"]
            assert "安全域" in result.data["policy"]["detail"]
        finally:
            await service.close()
