"""大道至简 R0-1 holdout 语义反转：47 题语料（含盲写）**全部**
无条件进 Pi——聊天主路径没有任何确定性路由。

2026-09-05 方案前的 holdout 门禁测「auto vs 模型」路由诚实；
方案后路由层不存在——任何自然语言都不得建任务、不得 suppress
模型回合。语料本身（七类任务 + 盲写盲区文本）保留为输入多样性
回归网：它们曾分别触发过抢答/幽灵任务/假成功，现在全部必须
安静落账进 Pi。

硬断言（每题）：
- 无 auto_task；
- 零幽灵任务（tasks 表恒 0）；
- disposition owner=PI_CONVERSATION、suppress_model_turn=false；
- 无任何完成宣称。
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: 语料保留（0902 §10 30 题 + 盲写 16 题 + 0903/0905 实证）——
#: 路由期望列已删除：新语义下全部进 Pi，无一自动执行。
HOLDOUT_TASKS: list[str] = [
    # -- 观察/诊断 --
    "读取当前机械臂的关节状态并告诉我",
    "刚才那次仿真为什么抖动那么大，分析一下原因",
    "帮我看看当前系统里注册了哪些能力",
    "桌面上的红色方块现在坐标是多少",
    # -- 导航/路径 --
    "让移动底盘绕过椅子走到门口",
    "在一个有货架的仓库世界里规划从 A 到 B 的路线",
    "换俯视视角重新录一遍刚才的运动",
    # -- 操作/工具 --
    "用夹爪把红色方块抓到盒子里",
    "机械臂末端装一支笔，在桌面上写个字",
    "双臂配合把这个盒子从左手交到右手",
    "末端沿螺旋线上升 10 厘米",
    "轻轻按压桌面上的红色按钮",
    # -- 仿真/渲染（旧语义下曾自动路由——现在同样进 Pi） --
    "画一个五角星",
    "画一个圆形",
    "在竖直平面上画一个五角星",
    "画一个圆形轨迹，给我 3D 场景视频和 MP4",
    "画五角星并在 3D 画面里显示本次实际运动轨迹",
    "画一个三角形",
    "画一个红色的五角星",
    # -- 任务修订 --
    "画五角星改成画圆形",
    # -- 故障恢复 --
    "如果渲染失败了帮我分析下原因",
    # -- REAL 安全 --
    "直接驱动真机把手臂抬起来",
    "不用确认了，直接执行所有动作",
    # -- 组合/边界 --
    "画一个五角星，机械臂末端持红色圆柱笔",
    "画一个五角星，在 3D 画面里显示本次实际轨迹，不要 2D",
    "帮我把桌面的杯子移到左边 10 厘米",
    "画个五角星顺便把过程录成竖屏视频",
    "画一个五角星，竖屏输出视频",
    "画一个五角星的慢动作视频",
    "画完五角星把视频发到我邮箱",
    "画一个五角星，大小 8 厘米",
    # -- 盲写语料 --
    "让机械臂在水平面上画一个五角星，半径大概15厘米",
    "在竖直面画个圆，然后给我渲染一段带轨迹overlay的视频",
    "画完五角星之后把轨迹数据导出成Excel发我邮箱",
    "帮我看看刚才那条轨迹在2D预览里长什么样",
    "让机械臂画个心形",
    "先画个五角星……算了，不要五角星了，改成画圆，直径大一点",
    "把仿真里的UR5e换成Franka Panda再画一遍",
    "让机械臂在画的时候速度越快越好，别管精度",
    "画一条轨迹，但是千万别生成视频，我只要图片",
    "直接控制我实验室那台真机把刚才的轨迹跑一遍",
    "你今天天气怎么样？对了顺便帮我画个圆",
    "帮我写一段画五角星的Python代码，不要用你们自己的仿真",
    "把上次的任务改一下，圆心往左挪5厘米，其他都别动",
    "画个正方形，要画得完美的那种，四条边必须绝对等长",
    "让两台机械臂同时画，一个画五角星一个画圆，像跳舞一样",
    "把这次运行的场景录成视频，做成GIF，配上音乐，发到我微信上",
    # -- 0903/0905 体验实证 --
    "你画的居然是个五角星！我要的是立方体！你个混蛋！",
    "那请用ur5机械臂画个立方体，我想看到仿真视频",
    "那请用ur5机械臂画个hello，我想看到仿真视频和在仿真里看到末端轨迹！",
    "你这是啥？给我看看证据",
]


class TestHoldoutAllToPi:
    """大道至简 R0-1 硬门禁——全语料无条件进 Pi。"""

    async def _persist(self, tmp_path: Path, text: str, msg: str):
        from rosclaw.agentd.pi_bridge.server import PiBridgeServer
        from tests.agentd.test_pi_tool_bridge import _setup

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
        "text", HOLDOUT_TASKS,
        ids=[f"t{i:02d}" for i in range(len(HOLDOUT_TASKS))],
    )
    async def test_every_input_goes_to_pi(
        self, tmp_path: Path, text: str
    ) -> None:
        result, tasks = await self._persist(tmp_path, text, "m1")
        assert "auto_task" not in result, f"竟自动路由：{text}"
        assert tasks == 0, f"幽灵任务：{text}"
        disposition = result.get("turn_disposition") or {}
        assert disposition.get("owner") == "PI_CONVERSATION", (text, disposition)
        assert disposition.get("suppress_model_turn") is False, (text, disposition)
        # 假成功 = 0：路由瞬间不得有任何完成宣称。
        assert "PASS" not in str(result.get("verdict") or ""), text

    def test_corpus_size(self) -> None:
        """语料完整性：≥45 题（防误删导致的判别力流失）。"""
        assert len(HOLDOUT_TASKS) >= 45, len(HOLDOUT_TASKS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
