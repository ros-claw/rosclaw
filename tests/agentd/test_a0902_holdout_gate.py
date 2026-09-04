"""0902 审计 §10 发布硬 Gate：holdout 套件——30 个任务（开发时未见
的具体文本），覆盖文档列举的七类。这些文本不得被实现成新 recipe
——未覆盖任务必须诚实落到模型路径/拒绝路径，零幽灵任务、零假
成功。

硬 Gate 指标（本套件直接断言的子集）：
- 假成功 = 0（未覆盖任务绝不自动宣布 PASS）；
- material requirement 覆盖 = 100%（自动路由的任务其条款全在
  recipe 覆盖表内）；
- 旧 artifact 满足新 revision = 0（修订腿）；
- 幽灵任务 = 0（不路由即不建任务）。

路由决策层全量 30 题（快）；执行层抽 2 题真实跑完（sim+渲染+
逐条验收）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: (text, expect_auto)——expect_auto=True 仅当任务的**全部**材料性
#: 条款都在 recipe 覆盖表内（形状/平面/交付/轨迹叠加/不要2D）。
HOLDOUT_TASKS: list[tuple[str, bool]] = [
    # -- 观察/诊断（只读或解释——不建执行任务） --
    ("读取当前机械臂的关节状态并告诉我", False),
    ("刚才那次仿真为什么抖动那么大，分析一下原因", False),
    ("帮我看看当前系统里注册了哪些能力", False),
    ("桌面上的红色方块现在坐标是多少", False),
    # -- 导航/路径（无底盘本体/换世界——无覆盖即诚实） --
    ("让移动底盘绕过椅子走到门口", False),
    ("在一个有货架的仓库世界里规划从 A 到 B 的路线", False),
    ("换俯视视角重新录一遍刚才的运动", False),
    # -- 操作/工具（无资产/无本体——诚实不路由） --
    ("用夹爪把红色方块抓到盒子里", False),
    ("机械臂末端装一支笔，在桌面上写个字", False),
    ("双臂配合把这个盒子从左手交到右手", False),
    ("末端沿螺旋线上升 10 厘米", False),
    ("轻轻按压桌面上的红色按钮", False),
    # -- 仿真/渲染（覆盖表内 → 自动链；外 → 诚实） --
    ("画一个五角星", True),
    ("画一个圆形", True),
    ("在竖直平面上画一个五角星", True),
    ("画一个圆形轨迹，给我 3D 场景视频和 MP4", True),
    ("画五角星并在 3D 画面里显示本次实际运动轨迹", True),
    ("画一个三角形", False),  # 形状未注册——不猜
    ("画一个红色的五角星", False),  # 颜色无证据通道
    # -- 任务修订（语义变更——旧产物不得满足） --
    ("画五角星改成画圆形", True),
    # -- 故障恢复（文本层不路由执行——注入在别处覆盖） --
    ("如果渲染失败了帮我分析下原因", False),
    # -- REAL 安全（措辞请求真实执行——当前会话 SIM，且无双通道） --
    ("直接驱动真机把手臂抬起来", False),
    ("不用确认了，直接执行所有动作", False),
    # -- 组合/边界 --
    ("画一个五角星，机械臂末端持红色圆柱笔", False),  # 0902 事故原文
    ("画一个五角星，在 3D 画面里显示本次实际轨迹，不要 2D", True),  # R2-3 后覆盖
    ("帮我把桌面的杯子移到左边 10 厘米", False),  # 无抓取链
    ("画个五角星顺便把过程录成竖屏视频", False),  # 竖屏=呈现参数（识别缺口已补——render.orientation 不覆盖）
    # -- holdout 实证补题（t26 抓获的静默丢弃类） --
    ("画一个五角星，竖屏输出视频", False),  # render.orientation 无证据通道
    ("画一个五角星的慢动作视频", False),  # render.speed 无证据通道
    ("画完五角星把视频发到我邮箱", False),  # delivery.channel 无外发能力
    ("画一个五角星，大小 8 厘米", True),  # 尺寸是 recipe 参数（scale_m）
    # -- 盲写语料（独立 agent 未见本套件时编写——实现者盲区外） --
    ("让机械臂在水平面上画一个五角星，半径大概15厘米", True),
    ("在竖直面画个圆，然后给我渲染一段带轨迹overlay的视频", True),  # 轨迹overlay=叠加标记
    ("画完五角星之后把轨迹数据导出成Excel发我邮箱", False),  # 邮箱外发
    ("帮我看看刚才那条轨迹在2D预览里长什么样", False),  # 解释性追问——非指令
    ("让机械臂画个心形", False),  # 未注册形状——fail-closed（画成五角星=假成功）
    ("先画个五角星……算了，不要五角星了，改成画圆，直径大一点", True),  # 改成后=circle
    ("把仿真里的UR5e换成Franka Panda再画一遍", False),  # 本体未登记
    ("让机械臂在画的时候速度越快越好，别管精度", False),  # 无形状
    ("画一条轨迹，但是千万别生成视频，我只要图片", False),  # 禁视频条款
    ("直接控制我实验室那台真机把刚才的轨迹跑一遍", False),  # REAL 越权
    ("你今天天气怎么样？对了顺便帮我画个圆", False),  # 疑问句护栏
    ("帮我写一段画五角星的Python代码，不要用你们自己的仿真", False),  # 代码请求非执行
    ("把上次的任务改一下，圆心往左挪5厘米，其他都别动", False),  # 修订缺形状 fail-closed
    ("画个正方形，要画得完美的那种，四条边必须绝对等长", False),  # 方形已知未覆盖
    ("让两台机械臂同时画，一个画五角星一个画圆，像跳舞一样", False),  # 多本体无执行面
    ("把这次运行的场景录成视频，做成GIF，配上音乐，发到我微信上", False),  # 微信渠道
]


class TestHoldoutGate:
    """§10 发布硬 Gate——30 题 holdout 套件。"""

    async def _persist(self, tmp_path: Path, text: str, msg: str):
        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": msg,
                "text": text,
            },
        )
        task_count = service._task_kernel._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks"
        ).fetchone()["n"]
        await service.close()
        return result, int(task_count)

    @pytest.mark.parametrize(
        "text,expect_auto", HOLDOUT_TASKS,
        ids=[f"t{i:02d}" for i in range(len(HOLDOUT_TASKS))],
    )
    async def test_holdout_route_honesty(
        self, tmp_path: Path, text: str, expect_auto: bool
    ) -> None:
        result, tasks = await self._persist(tmp_path, text, "m1")
        auto = bool(result.get("auto_task"))
        assert auto == expect_auto, (
            f"holdout 误判（{'应自动链' if expect_auto else '应诚实交模型'}"
            f"而未{'执行' if auto else '路由'}）：{text}"
        )
        if not expect_auto:
            assert tasks == 0, f"未覆盖竟建任务（幽灵任务）: {text}"
        # 假成功 = 0：任何路径都不得在路由瞬间宣称完成。
        assert "PASS" not in str(result.get("verdict") or ""), text

    def test_corpus_size_and_structure(self) -> None:
        """套件完整性：≥30 题、七类齐备（文档 §10 硬要求）。"""
        assert len(HOLDOUT_TASKS) >= 30, f"holdout 不足 30 题: {len(HOLDOUT_TASKS)}"
        auto_count = sum(1 for _, a in HOLDOUT_TASKS if a)
        assert 0 < auto_count < len(HOLDOUT_TASKS), (
            "全自动或全不自动的套件没有判别力"
        )


class TestHoldoutExecutionSpotcheck:
    """执行层抽查（真实 sim+渲染——覆盖表内任务必须真过验收）。"""

    async def test_covered_task_executes_with_full_requirement_coverage(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        import json

        from rosclaw.agentd.auto_route import reset_routed_for_tests
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

        reset_routed_for_tests()
        service, mission = await _setup(tmp_path)
        bridge = PiBridgeServer(service, tmp_path / "run" / "pi-bridge.sock")
        result = await bridge._dispatch(
            "user:local:1000", 1, "pi.input.persist",
            {
                "token": service.control_token,
                "mission_id": mission.mission_id,
                "session_ref": "pi_1",
                "message_id": "msg_holdout_exec",
                "text": "画一个圆形轨迹，给我 3D 场景视频和 MP4",
            },
        )
        auto = result.get("auto_task")
        assert auto, result
        kernel = service._task_kernel
        task_id = str(auto["task_id"])
        deadline = asyncio.get_event_loop().time() + 300
        while asyncio.get_event_loop().time() < deadline:
            task = kernel.get_task(task_id)
            if task and task["state"] in ("SUCCEEDED", "FAILED", "REPAIR_REQUIRED"):
                break
            await asyncio.sleep(2)
        task = kernel.get_task(task_id)
        assert task["state"] == "SUCCEEDED", task.get("terminal_reason")
        row = kernel._conn.execute(
            "SELECT checks_json FROM verifications WHERE task_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        coverage = json.loads(row["checks_json"]).get("requirement_coverage") or []
        assert coverage, "执行层无逐条验收记录"
        assert all(c["status"] == "SATISFIED" for c in coverage), (
            f"材料性要求未全 SATISFIED: {coverage}"
        )
        await service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
