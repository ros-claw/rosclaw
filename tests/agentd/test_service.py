"""AgentService / CLI / onboarding tests (PR-NA-040/041/042 exits).

- service starts with mock gateway, missions + turns over HTTP API
- duplicate start lock; REAL mode honestly refused with gap list
- onboarding writes api_key_ref only (never raw keys), MODEL_NOT_READY honesty
- console page served
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.cli import main as agentd_main
from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.context.sources import BodyFacts
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.onboarding import configure_model, doctor
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _answer_turn(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "dec_1",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t1",
        provider="mock",
        model="mock-model",
        content=f"你好，我是 ROSClaw。\n```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "你好"},
    )


@pytest.fixture
def service(tmp_path: Path) -> AgentService:
    # 七审 PR-SEVEN-1：本文件测 HTTP/API 面——禁第一方 kit（不起 MCP）。
    (tmp_path / "config.yaml").write_text(
        "agent:\n  enabled: true\nkits:\n  disabled: [rosclaw/ur5e-sim]\n",
        encoding="utf-8",
    )
    config = load_agent_config(tmp_path / "config.yaml")  # no config → defaults
    config.sim_body_id = "sim/ur5e"
    gateway = MockModelGateway(mock_profile(), [_answer_turn] * 50)
    # Make the script effectively unbounded: callable re-queues nothing.
    return AgentService(config, tmp_path, gateway=gateway)


class TestService:
    async def test_real_mode_refused_with_gaps(self, service: AgentService) -> None:
        with pytest.raises(Exception, match="prerequisites"):
            service.create_mission("搬箱子", mode="REAL")

    async def test_status_and_probe(self, service: AgentService) -> None:
        # PR-H9：无持久 gateway——status/probe 读配置真相；fixture 无
        # profiles 时诚实空面（不再从 mock gateway 拿 profile）。
        status = service.status()
        assert status["profile"] == ""
        probe = await service.probe()
        assert probe.reachable is False

    async def test_real_mission_binds_verified_live_body_and_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class LiveBodySource:
            def __init__(self, **_kwargs) -> None:
                pass

            def get_body(self, body_id: str) -> BodyFacts | None:
                if body_id != "limo":
                    return None
                return BodyFacts(
                    body_id="limo",
                    effective_body_hash="body_live_hash",
                    summary="verified live LIMO",
                    calibrated=True,
                )

        class LiveDaemon:
            def get_runtime_status(self) -> dict:
                return {
                    "running": True,
                    "runtime_state": "RUNNING",
                    "robot_id": "limo",
                    "robot_pack": {"loaded": True, "signature_status": "valid"},
                    "registered_executors": ["limo.play_tone:REAL"],
                }

        monkeypatch.setattr("rosclaw.agentd.service.ResolverBodySource", LiveBodySource)
        config = load_agent_config(tmp_path / "missing.yaml")
        config.body_id = "limo"
        config.default_mode = "REAL"
        config.physical_action_count = 3
        gateway = MockModelGateway(mock_profile(), [_answer_turn])
        live = AgentService(config, tmp_path, gateway=gateway)
        live._daemon_client = LiveDaemon()

        try:
            mission = live.create_mission("播放巡检提示音", mode="REAL")

            assert mission.body_binding.body_id == "limo"
            assert mission.body_binding.effective_body_hash == "body_live_hash"
            assert mission.mode.value == "REAL"
        finally:
            await live.close()

    async def test_daemon_consent_channel_is_wired_into_intent_handlers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class LiveBodySource:
            def __init__(self, **_kwargs) -> None:
                pass

            def get_body(self, body_id: str) -> BodyFacts | None:
                return BodyFacts(
                    body_id=body_id,
                    effective_body_hash="body_live_hash",
                    summary="verified live body",
                    calibrated=True,
                )

        monkeypatch.setattr("rosclaw.agentd.service.ResolverBodySource", LiveBodySource)
        socket_path = tmp_path / "run" / "rosclawd.sock"
        socket_path.parent.mkdir(parents=True)
        socket_path.touch()
        config = load_agent_config(tmp_path / "missing.yaml")
        config.body_id = "limo"
        live = AgentService(
            config,
            tmp_path,
            gateway=MockModelGateway(mock_profile(), [_answer_turn]),
        )
        try:
            assert live._consent_channel is not None
        finally:
            await live.close()


class TestHttpApi:
    @pytest.fixture
    def client(self, service: AgentService):
        from fastapi.testclient import TestClient

        return TestClient(create_app(service), headers={'x-rosclaw-token': service.control_token})

    def test_health_and_status(self, client) -> None:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/status").json()["maturity"] == "experimental"

    def test_real_mode_422_with_gaps(self, client) -> None:
        r = client.post("/missions", json={"goal": "搬箱子", "mode": "REAL"})
        assert r.status_code == 422
        assert "configured body is simulated" in r.json()["detail"]

    def test_console_served(self, client) -> None:
        r = client.get("/console")
        assert r.status_code == 200
        assert "ROSClaw Console" in r.text


class TestOnboarding:
    def test_loads_real_body_id_without_overwriting_legacy_sim_default(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "agent:\n  body_id: limo\n  default_mode: REAL\n"
            "  budgets:\n    physical_action_count: 3\n",
            encoding="utf-8",
        )
        config = load_agent_config(path)

        assert config.active_body_id == "limo"
        assert config.sim_body_id == "sim/ur5e"
        assert config.physical_action_count == 3

    def test_configure_writes_key_ref_only(self, tmp_path: Path) -> None:
        """P1-A1：setup 写 Pi 配置单源——key 只写 $ENV 引用。"""
        summary = configure_model(tmp_path, "kimi-code")
        assert summary["configured"]
        models = json.loads(
            (tmp_path / "agent" / "models.json").read_text(encoding="utf-8")
        )
        provider = models["providers"]["kimi-code"]
        assert provider["apiKey"] == "$ROSCLAW_KIMI_API_KEY"
        assert summary["api_key_ref"] == "env:ROSCLAW_KIMI_API_KEY"
        for path in (
            tmp_path / "agent" / "models.json",
            tmp_path / "agent" / "settings.json",
        ):
            assert "sk-" not in path.read_text(encoding="utf-8")

    def test_doctor_not_ready_without_key(self, tmp_path: Path, monkeypatch) -> None:
        configure_model(tmp_path, "kimi-code")
        monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
        report = doctor(tmp_path)
        assert report["status"] == "MODEL_NOT_READY"
        assert report["api_key_present"] is False

    def test_doctor_no_profiles(self, tmp_path: Path) -> None:
        report = doctor(tmp_path)
        assert report["status"] == "MODEL_NOT_READY"
        assert "setup model" in report["reason"]


class TestCli:
    def test_chat_requires_config(self, tmp_path: Path, capsys) -> None:
        rc = agentd_main(["--home", str(tmp_path), "chat"])
        assert rc == 2
        assert "setup model" in capsys.readouterr().err

    def test_status_json(self, tmp_path: Path, capsys) -> None:
        rc = agentd_main(["--home", str(tmp_path), "status"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["running"] is False

    def test_init_noninteractive_needs_provider(self, tmp_path: Path, capsys, monkeypatch) -> None:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        rc = agentd_main(["--home", str(tmp_path), "init"])
        assert rc == 2
