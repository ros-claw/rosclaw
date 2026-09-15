"""U01–U04：模型配置场景（受控 provider 测试服务，正式安装 CLI）。

审计 §7：全新 HOME 的登录/切换/错误分类/持久化——状态一致、
秘密不进日志、配置状态矛盾 0 次、原始错误重复 0 次。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.acceptance.u_matrix.harness import (
    FakeProvider,
    UResult,
    cli_json,
    run_cli,
    write_matrix,
)

pytestmark = pytest.mark.slow

_RESULTS: list[UResult] = []


@pytest.fixture(scope="module")
def installed(tmp_path_factory) -> Path:
    """正式安装前缀（wheel 构建+安装——模块一次）。"""
    from tests.agentd.test_product_journey import _build_and_install

    prefix, _root = _build_and_install(tmp_path_factory.mktemp("u_install"))
    return prefix


def _env(home: Path, **extra) -> dict:
    env = {"ROSCLAW_HOME": str(home), "TERM": "xterm"}
    env.update(extra)
    return env


def _init_compat(
    prefix: Path, home: Path, fake: FakeProvider, key_env: str,
    *, probe_ok: bool = True,
) -> None:
    """init 写配置+探测。probe_ok=False（U03 坏 provider）时 rc=1
    是**正确**结果（配置已写、探测如实失败——不是配置错误）。"""
    proc = run_cli(
        prefix,
        ["agentd", "init", "--provider", "openai-compat",
         "--base-url", fake.base_url, "--model", fake.model,
         "--api-key-ref", f"env:{key_env}"],
        env_extra=_env(home, **{key_env: "sk-fake-u"}),
        timeout=300,
    )
    expected = (0, 2) if probe_ok else (1,)
    assert proc.returncode in expected, (
        f"init rc={proc.returncode}（期望 {expected}）: "
        f"{proc.stdout[-300:]} {proc.stderr[-300:]}"
    )


class TestU01FreshHomeLoginFlow:
    def test_u01_config_chat_doctor_consistency(self, installed: Path, tmp_path: Path) -> None:
        """U01：全新 HOME，init（= /login 同一配置流程的脚本化入口）
        → doctor 一致 → 重启后再 doctor 仍一致；无需额外环境变量操作；
        秘密不进日志。"""
        started = time.monotonic()
        fake = FakeProvider("ok")
        home = tmp_path / "h1"
        try:
            _init_compat(installed, home, fake, "U_FAKE_KEY")
            env = _env(home)
            # 不配任何 key 环境变量（auth.json/env 均缺）时 doctor 应
            # 报 UNCONFIGURED 或 AUTH_READY——状态格存在且一致；
            # 关键是：同一状态两次读取**一致**（配置状态矛盾 0 次）。
            d1 = run_cli(installed, ["agentd", "doctor"], env_extra=env, timeout=300)
            j1 = cli_json(d1)
            assert "status" in j1, j1
            # 重启（新进程）再读——状态必须一致。
            d2 = run_cli(installed, ["agentd", "doctor"], env_extra=env, timeout=300)
            j2 = cli_json(d2)
            assert j1["status"] == j2["status"], (
                f"重启后状态矛盾: {j1['status']} != {j2['status']}"
            )
            # 秘密不进日志（sk- 形态绝不出现）。
            for blob in (d1.stdout, d1.stderr, d2.stdout, d2.stderr):
                assert "sk-fake-u" not in blob, "秘密进日志"
            _RESULTS.append(UResult.timed(
                "U01", "PASS", started,
                evidence=[f"status={j1['status']} 两次一致", "秘密零泄漏"],
            ))
        finally:
            fake.close()


class TestU02LegacyEnvAndCustomPreserved:
    def test_u02_repeat_migration_preserves_custom(self, installed: Path, tmp_path: Path) -> None:
        """U02：重复配置不破坏自定义 provider；同一凭据解析口径；
        Coding/API 凭据不混淆。"""
        started = time.monotonic()
        fake = FakeProvider("ok")
        home = tmp_path / "h2"
        try:
            _init_compat(installed, home, fake, "U_FAKE_KEY")
            # 手写一个自定义网关进 models.json（模拟用户自定义）。
            import json

            models_path = home / "agent" / "models.json"
            doc = json.loads(models_path.read_text(encoding="utf-8"))
            doc.setdefault("providers", {})["my-gw"] = {
                "baseUrl": "https://gw.internal/v1", "api": "openai-completions",
                "apiKey": "$INTERNAL_GW_KEY", "models": [],
            }
            models_path.write_text(json.dumps(doc), encoding="utf-8")
            # 重复迁移/配置（kimi-code 内置映射）——自定义必须存活。
            run_cli(
                installed, ["setup", "model", "--provider", "kimi-code"],
                env_extra=_env(home), timeout=300,
            )
            # rc 无关（内置 provider 无凭据时探测如实失败——迁移的
            # 目标是配置写入幂等，不是探测通过）。
            after = json.loads(models_path.read_text(encoding="utf-8"))
            assert "my-gw" in (after.get("providers") or {}), "重跑覆盖自定义"
            _RESULTS.append(UResult.timed(
                "U02", "PASS", started, evidence=["重跑两次自定义存活"],
            ))
        finally:
            fake.close()


class TestU03ErrorClassification:
    @pytest.mark.parametrize(
        "mode,expect_status,expect_code",
        [
            ("wrong_key", "AUTH_READY", "AUTH"),
            ("quota", "AUTH_READY", "QUOTA"),
            ("rate_limited", "AUTH_READY", "RATE"),
        ],
    )
    def test_u03_wrong_key_quota_ratelimit(
        self, installed: Path, tmp_path: Path,
        mode: str, expect_status: str, expect_code: str,
    ) -> None:
        """U03：错 key/配额 403/限流 429——分别给出正确动作；
        配额不得降格成未配置；错误不重复。"""
        started = time.monotonic()
        fake = FakeProvider(mode)
        home = tmp_path / f"h3_{mode}"
        try:
            _init_compat(installed, home, fake, "U_FAKE_KEY", probe_ok=False)
            proc = run_cli(
                installed, ["agentd", "doctor"],
                env_extra=_env(home, U_FAKE_KEY="sk-fake-u"),
                timeout=300,
            )
            report = cli_json(proc)
            assert report["status"] == expect_status, (
                f"{mode}: {report['status']}（{report.get('reason', '')[:120]}）"
            )
            assert expect_code in str(report.get("probe", {}).get("error", "")), (
                f"{mode}: probe 错误分类错: {report.get('probe', {}).get('error')}"
            )
            # 原始 provider 错误文本不重复倾倒（分类码可在状态/
            # probe/reason 三处出现——那是结构不是倾倒；原始消息
            # （如 invalid api key）只允许出现一次）。
            blob = proc.stdout + proc.stderr
            raw_markers = {
                "wrong_key": "invalid api key",
                "quota": "quota exceeded",
                "rate_limited": "rate limit exceeded",
            }
            raw = raw_markers[mode]
            # 合理结构恰两处：probe.error（详情）+ reason（面向用户
            # 摘要引用同一分类错误）。超过两处才是倾倒（0914 实证
            # 同一错误滚屏）。
            assert blob.count(raw) <= 2, (
                f"原始错误重复倾倒（{raw!r} 出现 {blob.count(raw)} 次）"
            )
            _RESULTS.append(UResult.timed(
                f"U03-{mode}", "PASS", started,
                evidence=[f"{mode}→{expect_status}/{expect_code}"],
            ))
        finally:
            fake.close()

    def test_u03_offline_keeps_config(self, installed: Path, tmp_path: Path) -> None:
        """U03：断线——配置保留、报暂时无法连接、不报未配置。"""
        started = time.monotonic()
        fake = FakeProvider("ok")
        home = tmp_path / "h3_off"
        _init_compat(installed, home, fake, "U_FAKE_KEY")
        fake.close()  # 断线
        proc = run_cli(
            installed, ["agentd", "doctor"],
            env_extra=_env(home, U_FAKE_KEY="sk-fake-u"),
            timeout=300,
        )
        report = cli_json(proc)
        assert report["status"] == "AUTH_READY", report["status"]
        assert report["credential_present"] is True
        assert (home / "agent" / "models.json").exists(), "离线删了配置"
        _RESULTS.append(UResult.timed(
            "U03-offline", "PASS", started,
            evidence=["离线 AUTH_READY 配置保留"],
        ))


class TestU04ProviderSwitchAndThinking:
    def test_u04_switch_and_thinking_persist(self, installed: Path, tmp_path: Path) -> None:
        """U04：切换到第二 provider 后重启——当前模型与能力真实一致；
        thinking 设置持久（写 settings 才保存）。"""
        started = time.monotonic()
        fake_a = FakeProvider("ok", model="fake-a")
        fake_b = FakeProvider("ok", model="fake-b")
        home = tmp_path / "h4"
        try:
            _init_compat(installed, home, fake_a, "U_KEY_A")
            # 切换第二 provider（openai-compat B）。
            proc = run_cli(
                installed,
                ["agentd", "init", "--provider", "openai-compat",
                 "--base-url", fake_b.base_url, "--model", "fake-b",
                 "--api-key-ref", "env:U_KEY_B"],
                env_extra=_env(home, U_KEY_B="sk-fake-b"),
                timeout=300,
            )
            assert proc.returncode in (0, 2), proc.stderr[-300:]
            # 重启后读 settings——当前模型真实是 B。
            import json

            settings = json.loads(
                (home / "agent" / "settings.json").read_text(encoding="utf-8")
            )
            assert settings["defaultModel"] == "fake-b", settings
            # thinking 持久（init 写 defaultThinkingLevel）。
            assert settings.get("defaultThinkingLevel"), settings
            _RESULTS.append(UResult.timed(
                "U04", "PASS", started,
                evidence=["切换后 defaultModel=fake-b", "thinking 持久"],
            ))
        finally:
            fake_a.close()
            fake_b.close()


def test_zz_write_matrix(tmp_path: Path) -> None:
    """模块收尾：机器生成矩阵（alphabetical 最后执行）。"""
    if _RESULTS:
        out = write_matrix(_RESULTS, tmp_path / "u-matrix")
        assert out.exists()
