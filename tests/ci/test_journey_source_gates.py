"""PR-SEVEN-7 红测试（七审 §6 CI 结构门禁）：Journey 证据 scope。

红测试先行——当前缺陷：

1. Product Journey 安装后仍能看到源码 checkout——"clean-install" 证据
   可以被仓库源码路径喂绿（必须在 journey 运行期间把 checkout 改名，
   并在证据里记录 source_checkout_accessible=false）；
2. journey 源码层禁令没有 CI 门禁（手写 mcp_servers / 引用仓库
   executor 的回归可以悄悄混进旅程）；
3. 三旅程只有 A——缺 Journey B（ask-every-time）与 Journey C
   （REAL hard boundary）；
4. evidence manifest 缺 install_origin/config_origin/robot_kit_digest/
   source_checkout_accessible；独立 verifier 不校验 journey scope。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOURNEY = REPO / "tests" / "agentd" / "test_product_journey.py"
MANIFEST_SH = REPO / "scripts" / "ci" / "write_evidence_manifest.sh"
VERIFIER = REPO / "scripts" / "ci" / "verify_journey_evidence.py"


class TestJourneySourceHygiene:
    def test_no_handwritten_mcp_servers(self) -> None:
        source = JOURNEY.read_text(encoding="utf-8")
        assert "mcp_servers" not in source, (
            "journey 源码出现 mcp_servers——UR5e 能力必须来自发行包第一方 "
            "Robot Kit 自动激活，不是测试夹具注入"
        )

    def test_no_repo_source_executor_refs(self) -> None:
        """禁止把仓库源码路径当安装产物 executor 引用。"""
        source = JOURNEY.read_text(encoding="utf-8")
        for pattern in (
            r'REPO\s*/\s*"src',
            r"src/rosclaw/sim/\w+_mcp\.py",
            r"src/rosclaw/limo/\w+_mcp\.py",
        ):
            assert not re.search(pattern, source), (
                f"journey 源码引用仓库 executor 路径: {pattern}"
            )

    def test_checkout_hidden_during_journey(self) -> None:
        """journey 运行期间源码 checkout 必须不可达（改名/撤权），且
        证据记录 source_checkout_accessible=false。"""
        source = JOURNEY.read_text(encoding="utf-8")
        assert "source_checkout_accessible" in source, (
            "journey 未记录 source_checkout_accessible"
        )
        # 必须有真实的隐藏动作（rename/chmod），不只是写个字段。
        assert re.search(r"(os\.rename|\.rename\(|chmod)", source), (
            "journey 没有实际隐藏源码 checkout 的动作"
        )


class TestThreeJourneysExist:
    def test_journey_class_keeps_slow_mark(self) -> None:
        """journey 类必须带 @pytest.mark.slow——否则 Full Regression
        （无 .venv 的 uv --system 环境）会收进旅程并全线失败
        （PR-SEVEN-7 首次 CI 实测回归：装饰器被重构留在 helper 上）。"""
        source = JOURNEY.read_text(encoding="utf-8")
        assert "@pytest.mark.slow\nclass TestProductJourney" in source, (
            "@pytest.mark.slow 不在 TestProductJourney 类上"
        )

    def test_journey_b_ask_every_time_exists(self) -> None:
        source = JOURNEY.read_text(encoding="utf-8")
        assert "def test_journey_b_ask_every_time" in source, (
            "缺 Journey B（SIM ask-every-time：一键 Operator 初始化 + "
            "一次人工卡覆盖整条轨迹 + deny fail closed）"
        )

    def test_journey_c_real_boundary_exists(self) -> None:
        source = JOURNEY.read_text(encoding="utf-8")
        assert "def test_journey_c_real_boundary" in source, (
            "缺 Journey C（REAL hard boundary：SIM grant 不跨 REAL、"
            "缺真机/presence/permit 不可建卡执行）"
        )


class TestManifestAndVerifierScope:
    def test_manifest_has_scope_fields(self) -> None:
        source = MANIFEST_SH.read_text(encoding="utf-8")
        for field in (
            "install_origin",
            "config_origin",
            "robot_kit_digest",
            "source_checkout_accessible",
        ):
            assert field in source, f"evidence manifest 缺 {field}"

    def test_verifier_checks_journey_scope(self) -> None:
        source = VERIFIER.read_text(encoding="utf-8")
        assert "journey_scope" in source, "独立 verifier 不校验 journey_scope"
        assert "source_checkout_accessible" in source
