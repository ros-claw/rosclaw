"""十四审 PR-14.7 红测试：产品清洁度（总纲 §1.9/§7 PR-14.7）。

红测试先行——修复前必须红：
1. delegate 摘要不得把 soft target 显示成 "Deadline"（概念误导——
   Native Agent 据此错误归因"300 秒墙钟强杀"）；只有显式权威的
   hard deadline 才允许叫 Deadline；
2. 启动诊断路由覆盖所有 AgentService 入口（agentd server 入口也
   必须安装）；pydantic 2.13 forward-ref 第三方告警按 类别+消息
   定向过滤（不是全局屏蔽，其余 warning 仍进日志）；
3. 启动导入链在 warnings-as-errors 下必须干净（防回归）。
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
from tests.agentd.test_pi_tool_bridge import _request, _setup


class TestDeadlineLabel:
    async def test_delegate_summary_no_fake_deadline(self, tmp_path: Path) -> None:
        """soft target 是'预计/提醒阈值'，不是 Deadline（§1.9）。"""
        service, mission = await _setup(tmp_path)
        dispatcher = PiToolDispatcher(service)
        # 慢 worker 强制走 STARTED 摘要路径（快任务会同步完成）。
        from tests.agentd.test_ten_w0 import _register_stub, _slow_adapter_module

        stub = _slow_adapter_module()()
        _register_stub(
            service, stub, worker_id="worker:stub:slow",
            adapter_type="process_stdio",
        )
        result = await dispatcher.execute(
            _request(
                "rosclaw_delegate",
                mission=mission.mission_id,
                idem="idem_147_label",
                arguments={"goal": "慢任务", "worker_id": "worker:stub:slow"},
            )
        )
        assert result.ok and result.status == "STARTED", result.summary
        assert "Deadline" not in result.summary, result.summary
        assert "提醒阈值" in result.summary or "预计" in result.summary
        wo = result.summary.split("WorkOrder: ")[1].split("\n")[0]
        await dispatcher.execute(
            _request(
                "rosclaw_cancel_work",
                mission=mission.mission_id,
                idem="idem_147_cancel",
                arguments={"work_order_id": wo},
            )
        )
        await service.close()


class TestWarningsRouting:
    def test_incomplete_definition_filtered_by_category_and_message(
        self, tmp_path: Path
    ) -> None:
        """pydantic 2.13 第三方 forward-ref 告警：按类别+消息定向过滤
        （任何模块来源）；其他 warning 不受影响。"""
        from rosclaw.agentd.cli import _route_internal_diagnostics_to_log

        _route_internal_diagnostics_to_log(tmp_path, debug=False)
        # 不重置过滤器（simplefilter 会覆盖路由安装的 ignore 规则——
        # 真实启动路径没有 simplefilter 调用）。
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn(
                "Field `model_dump_json` has an incomplete definition",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn(
                "some other warning stays visible", UserWarning, stacklevel=2
            )
        messages = [str(w.message) for w in caught]
        assert not any("incomplete definition" in m for m in messages)
        assert any("some other warning" in m for m in messages)

    def test_agentd_server_entry_installs_routing(self) -> None:
        """agentd server 入口（cli.py 的 serve 路径）也必须先装诊断
        路由再构造 AgentService——十三审只覆盖了 chat 入口（§1.9
        干净安装入口仍漏）。"""
        import inspect

        from rosclaw.agentd import cli

        serve_src = inspect.getsource(cli.cmd_start)
        assert "_route_internal_diagnostics_to_log" in serve_src, (
            "agentd serve 入口（cmd_start）必须在 AgentService 前安装诊断路由"
        )
        assert serve_src.index("_route_internal_diagnostics_to_log") < serve_src.index(
            "AgentService(config, home)"
        )

    def test_startup_imports_warning_clean(self) -> None:
        """启动导入链在 warnings-as-errors 下必须干净（根因防线——
        任何未来依赖引入定义期告警都会在本测试爆炸）。"""
        proc = subprocess.run(
            [
                sys.executable,
                "-W", "error::UserWarning",
                "-c",
                "import rosclaw.agentd.cli\n"
                "import rosclaw.agentd.service\n"
                "import rosclaw.contracts.export\n"
                "print('clean')",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        assert "clean" in proc.stdout
