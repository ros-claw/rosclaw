"""0914 PR-1：凭据诊断单源化——doctor 不再 env 直读短路。

实证根因（onboarding.py 旧码）：``if not key and model.api_key_ref:
UNCONFIGURED``——Python 直接读环境变量一票否决 Pi probe 的真实
解析结果。用户在 chat 内 /login（auth.json）后 chat 可用、doctor
仍报 UNCONFIGURED；Kimi 配额 403 被降格成"未配置"。

契约（审计 §3.3）：凭据存在、服务端认证、配额、工具调用分别报告；
凭据文件存在只算已配置；403 quota 不降格为未配置；离线不删配置；
tool probe 失败不覆盖 chat 事实。
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd import onboarding
from rosclaw.agentd.models.gateway import ModelProbeResult
from rosclaw.agentd.onboarding import doctor


async def _async_result(value):
    return value


def _probe(**kw) -> ModelProbeResult:
    base = {
        "reachable": True,
        "models_visible": ("kimi-for-coding",),
        "expected_model_present": True,
        "chat_ok": True,
        "tool_call_ok": True,
        "auth_configured": True,
    }
    base.update(kw)
    return ModelProbeResult(**base)


def _configured_home(home: Path, monkeypatch, probe: ModelProbeResult) -> None:
    """写好 Pi 配置（内置 kimi-coding——无 env 依赖）+ mock probe。"""
    onboarding.configure_model(home, "kimi-code")
    monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setattr(
        onboarding, "probe_home",
        lambda home, *, deep=False: _async_result(probe),
    )


class TestCredentialUnifiedDiagnosis:
    def test_login_only_no_env_is_chat_ready(self, tmp_path: Path, monkeypatch) -> None:
        """chat 内 /login（auth.json 有凭据）、无 env——Pi 解析成功
        即已配置；不得被 env 直读判成 UNCONFIGURED（旧码红）。"""
        _configured_home(tmp_path, monkeypatch, _probe())
        # auth.json 模拟 /login 结果（内容无关紧要——doctor 只认
        # probe 的 auth_configured，不解析凭据文件）。
        auth_dir = tmp_path / "agent"
        auth_dir.mkdir(parents=True, exist_ok=True)
        (auth_dir / "auth.json").write_text(
            json.dumps({"kimi-coding": {"type": "api_key"}}), encoding="utf-8"
        )
        report = doctor(tmp_path)
        assert report["status"] in ("CHAT_READY", "TOOL_READY"), (
            f"/login 用户被 env 直读误判: {report['status']} — {report.get('reason')}"
        )
        assert report["credential_present"] is True

    def test_quota_403_is_configured_with_quota_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """配额耗尽（服务商事实）不得降格成未配置——AUTH_READY +
        quota_state=exhausted + 指引 /model 换已配置模型。"""
        _configured_home(
            tmp_path, monkeypatch,
            _probe(
                chat_ok=False, tool_call_ok=False,
                error="QUOTA_EXHAUSTED: 403 quota exceeded",
            ),
        )
        report = doctor(tmp_path)
        assert report["status"] != "UNCONFIGURED", report
        assert report["status"] == "AUTH_READY"
        assert report["quota_state"] == "exhausted"
        assert "/model" in report["reason"]
        assert report["credential_present"] is True

    def test_auth_failed_is_configured_but_rejected(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """401/403（非配额）= 凭据已配置但服务端拒绝——仍不是
        未配置（用户不需要重新 /login 以外的误导动作）。"""
        _configured_home(
            tmp_path, monkeypatch,
            _probe(
                chat_ok=False, tool_call_ok=False,
                error="AUTH_FAILED: 401 unauthorized",
            ),
        )
        report = doctor(tmp_path)
        assert report["status"] == "AUTH_READY"
        assert report["quota_state"] in ("unknown", "ok")
        assert "拒绝" in report["reason"]

    def test_no_credential_anywhere_is_unconfigured(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Pi 解析也找不到凭据 = 真 UNCONFIGURED（指向登录动作）。"""
        _configured_home(
            tmp_path, monkeypatch,
            _probe(
                reachable=False, chat_ok=False, tool_call_ok=False,
                auth_configured=False,
                error="AUTH_NOT_CONFIGURED: provider kimi-coding 无可用凭据",
            ),
        )
        report = doctor(tmp_path)
        assert report["status"] == "UNCONFIGURED"
        assert report["credential_present"] is False

    def test_offline_keeps_config(self, tmp_path: Path, monkeypatch) -> None:
        """离线/不可达 = 已配置但暂时无法连接——不得报未配置，
        不得暗示删配置。"""
        _configured_home(
            tmp_path, monkeypatch,
            _probe(
                reachable=False, chat_ok=False, tool_call_ok=False,
                error="NETWORK_UNREACHABLE: fetch failed",
            ),
        )
        report = doctor(tmp_path)
        assert report["status"] == "AUTH_READY"
        assert report["credential_present"] is True
        assert "无法连接" in report["reason"] or "不可达" in report["reason"]


class TestBuiltinKimiCodingDefault:
    def test_setup_writes_builtin_kimi_coding(self, tmp_path: Path) -> None:
        """setup 默认映射到 Pi 内置 kimi-coding（/login 原生可用）——
        settings 指向内置 provider；不再写自定义 provider 目录条目
        冒充内置服务。"""
        summary = onboarding.configure_model(tmp_path, "kimi-code")
        assert summary["configured"] is True
        settings = json.loads(
            (tmp_path / "agent" / "settings.json").read_text(encoding="utf-8")
        )
        assert settings["defaultProvider"] == "kimi-coding"
        assert settings["defaultModel"]
        models_path = tmp_path / "agent" / "models.json"
        # 内置 provider 不需要 models.json 条目——文件可不存在；
        # 若存在且含 kimi-coding 条目，endpoint 不得偏离官方服务
        # （防自定义漂移冒充官方）。
        models_doc = (
            json.loads(models_path.read_text(encoding="utf-8"))
            if models_path.exists() else {}
        )
        custom = (models_doc.get("providers") or {}).get("kimi-coding")
        if custom is not None:
            assert "api.kimi.com/coding" in str(custom.get("baseUrl", ""))

    def test_setup_rerun_preserves_custom_providers(self, tmp_path: Path) -> None:
        """迁移重跑不覆盖用户自定义 provider（审计 §3.6）。"""
        onboarding.configure_model(tmp_path, "kimi-code")
        models_path = tmp_path / "agent" / "models.json"
        models_path.parent.mkdir(parents=True, exist_ok=True)
        doc = (
            json.loads(models_path.read_text(encoding="utf-8"))
            if models_path.exists() else {}
        )
        doc.setdefault("providers", {})["my-gateway"] = {
            "baseUrl": "https://gw.internal/v1", "api": "openai-completions",
            "apiKey": "$INTERNAL_GW_KEY", "models": [],
        }
        models_path.write_text(json.dumps(doc), encoding="utf-8")
        onboarding.configure_model(tmp_path, "kimi-code")
        after = json.loads(models_path.read_text(encoding="utf-8"))
        assert "my-gateway" in (after.get("providers") or {})
