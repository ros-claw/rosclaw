"""Fixture corpus for the regime benchmark (数据库优化v4 §10, PR-BENCH-4).

Session-holdout corpus anchored to REAL machine statistics:

* ``sess_dev_cold_01`` / ``sess_dev_cold_02`` (dev): 36–50 °C healthy
  sessions (real: prac_20260715T140136Z ran 36–38 °C).
* ``sess_test_hot_01`` (test holdout): 56–58 °C two-hour thermal
  degradation (the PR #98 run1 regime).
* ``sess_test_warm_01`` (test holdout): 48–50 °C short healthy session
  (the PR #98 run2 no-memory arm regime).

Ground truth is PHYSICAL, not matcher-derived: a memory authored inside a
hot-session envelope is inapplicable to a cold-regime query BY
CONSTRUCTION.  Labels never come from the matcher under test.
"""

from __future__ import annotations

from typing import Any

DEV_SESSIONS = ("sess_dev_cold_01", "sess_dev_cold_02")
TEST_SESSIONS = ("sess_test_hot_01", "sess_test_warm_01")

# Regime contexts a query can be posed in (used by the runner to build the
# OperatingRegime the matcher/pipeline sees).
REGIME_CONTEXTS: dict[str, dict[str, Any]] = {
    "cold": {
        "regime_label": "COLD_HEALTHY",
        "temperature_c": 38.0,
        "temperature_slope_c_per_min": 0.01,
        "session_elapsed_sec": 480.0,
        "cumulative_action_count": 60,
        "position_error_p95": 5.0,
        "recent_failure_rate": 0.0,
        "recent_invalid_rate": 0.02,
    },
    "warm": {
        "regime_label": "WARM_STABLE",
        "temperature_c": 49.5,
        "temperature_slope_c_per_min": 0.05,
        "session_elapsed_sec": 900.0,
        "cumulative_action_count": 120,
        "position_error_p95": 8.0,
        "recent_failure_rate": 0.01,
        "recent_invalid_rate": 0.04,
    },
    "hot": {
        "regime_label": "THERMAL_TRACKING_DEGRADATION",
        "temperature_c": 57.0,
        "temperature_slope_c_per_min": 0.25,
        "session_elapsed_sec": 5400.0,
        "cumulative_action_count": 900,
        "position_error_p95": 22.0,
        "recent_failure_rate": 0.12,
        "recent_invalid_rate": 0.10,
    },
}


def _memory(
    memory_id: str,
    *,
    session: str,
    body: str,
    joint: str | None,
    failure_type: str,
    title: str,
    document: str,
    hint: str,
    category: str,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "memory_type": "failure",
        "robot_id": "rh56_rps_robot",
        "session_id": session,
        "body_id": body,
        "joint_name": joint,
        "failure_type": failure_type,
        "title": title,
        "document": document,
        "outcome": "failure",
        "category": category,
        "metadata": {"recovery_hint": hint},
        "evidence_refs": [f"{session}:evt_{memory_id}"],
    }


def _envelope(
    memory_id: str,
    *,
    body: str,
    regime_label: str,
    envelope_type: str,
    temp: tuple[float | None, float | None] = (None, None),
    elapsed: tuple[float | None, float | None] = (None, None),
    joints: list[str] | None = None,
    confidence: float = 0.8,
    evidence_count: int = 4,
    success_count: int = 3,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "body_ids": [body],
        "task_ids": ["rh56_rps"],
        "joints": joints or [],
        "temperature_min": temp[0],
        "temperature_max": temp[1],
        "temperature_slope_min": None,
        "temperature_slope_max": None,
        "elapsed_sec_min": elapsed[0],
        "elapsed_sec_max": elapsed[1],
        "regime_labels": [regime_label] if regime_label else [],
        "envelope_type": envelope_type,
        "evidence_count": evidence_count,
        "success_count": success_count,
        "confidence": confidence,
        "required_features": [],
        "reason": reason,
    }


def corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(memories, envelopes) for the session-holdout fixture."""
    memories = [
        # --- hot session (test holdout): thermal degradation failures ---
        _memory(
            "mem_hot_middle_jnr",
            session="sess_test_hot_01",
            body="rh56_right_01",
            joint="middle",
            failure_type="joint_not_reached",
            title="右手 middle 热退化下未达到目标位置",
            document="两小时 56–58°C 会话中 middle joint_not_reached，位置误差随温度上升；记录为冷却间隔恢复",
            hint="增加回合间冷却",
            category="rh56",
        ),
        _memory(
            "mem_hot_thumb_jnr",
            session="sess_test_hot_01",
            body="rh56_right_01",
            joint="thumb",
            failure_type="joint_not_reached",
            title="右手 thumb 热漂移未到位",
            document="57°C 附近 thumb joint_not_reached，热漂移导致角度偏差增大",
            hint="增加回合间冷却",
            category="rh56",
        ),
        # --- cold dev sessions: same failure type, healthy regime ---
        _memory(
            "mem_cold_middle_jnr",
            session="sess_dev_cold_01",
            body="rh56_right_01",
            joint="middle",
            failure_type="joint_not_reached",
            title="右手 middle 偶发未到位（低温健康工况）",
            document="36–38°C 短会话 middle 偶发 joint_not_reached，与标定边界有关，与温度无关",
            hint="检查标定零位",
            category="rh56",
        ),
        _memory(
            "mem_cold2_index_jnr",
            session="sess_dev_cold_02",
            body="rh56_right_01",
            joint="index",
            failure_type="joint_not_reached",
            title="右手 index 未到位（标定后）",
            document="标定后 index 偶发 joint_not_reached，低温健康工况",
            hint="复查 index 标定",
            category="rh56",
        ),
        # --- cross-body hard negative ---
        _memory(
            "mem_left_middle_jnr",
            session="sess_dev_cold_01",
            body="rh56_left_01",
            joint="middle",
            failure_type="joint_not_reached",
            title="左手 middle 未到位",
            document="左手 middle joint_not_reached，左手 OK 几何接触异常",
            hint="检查左手接触几何",
            category="rh56",
        ),
        # --- cross-task categories ---
        _memory(
            "mem_d435i_wedge",
            session="sess_dev_cold_01",
            body="d435i_01",
            joint=None,
            failure_type="camera_wedge",
            title="D435i 管道启动 wedge（UVC GET_CUR -110）",
            document="RealSense D435i 在固件状态异常时 pipe.start() 卡住，UVC GET_CUR -110，需要 hardware_reset",
            hint="librealsense hardware_reset，勿 sysfs 复位",
            category="d435i",
        ),
        _memory(
            "mem_ch340_eio",
            session="sess_dev_cold_01",
            body="rh56_left_01",
            joint=None,
            failure_type="serial_eio",
            title="CH340 适配器持续 EIO",
            document="CH340 USB-RS485 在重启后仍 Input/output error，需更换或物理重插",
            hint="物理重插或更换适配器",
            category="ch340_ftdi",
        ),
        _memory(
            "mem_lerobot_sync",
            session="sess_dev_cold_02",
            body="rh56_right_01",
            joint=None,
            failure_type="dataset_sync_lag",
            title="LeRobot 数据集会话同步滞后",
            document="lerobot 数据集会话语义与实际片段不一致，重新导出后恢复",
            hint="重新导出数据集",
            category="lerobot",
        ),
        _memory(
            "mem_chassis_slip",
            session="sess_dev_cold_02",
            body="chassis_01",
            joint=None,
            failure_type="wheel_slip",
            title="移动底盘低速打滑",
            document="移动底盘在光滑地面低速启动打滑，降低加速度后恢复",
            hint="降低启动加速度",
            category="chassis",
        ),
        _memory(
            "mem_vln_lost",
            session="sess_dev_cold_02",
            body="vln_01",
            joint=None,
            failure_type="navigation_lost",
            title="VLN 长走廊定位丢失",
            document="视觉语言导航在长走廊特征缺失段落定位丢失，回退最近航点",
            hint="回退最近航点重定位",
            category="vln",
        ),
        _memory(
            "mem_sandbox_crash",
            session="sess_dev_cold_01",
            body="sandbox_01",
            joint=None,
            failure_type="sim_crash",
            title="沙箱接触求解崩溃",
            document="MuJoCo 沙箱在高穿透接触下求解器崩溃，降低接触刚度后恢复",
            hint="降低接触刚度参数",
            category="sandbox",
        ),
        _memory(
            "mem_provider_oom",
            session="sess_dev_cold_02",
            body="provider_01",
            joint=None,
            failure_type="provider_oom",
            title="Embedding Provider 显存溢出",
            document="本地 embedding provider 在并发请求下显存溢出，限流后恢复",
            hint="限制并发请求数",
            category="provider",
        ),
        _memory(
            "mem_gpu_throttle",
            session="sess_test_hot_01",
            body="gpu_01",
            joint=None,
            failure_type="gpu_throttle",
            title="GPU 服务热降频",
            document="Jetson GPU 在高温环境热降频，推理延迟上升，改善散热后恢复",
            hint="改善散热或降低推理频率",
            category="gpu",
        ),
    ]
    envelopes = [
        _envelope(
            "mem_hot_middle_jnr",
            body="rh56_right_01",
            regime_label="THERMAL_TRACKING_DEGRADATION",
            envelope_type="validated",
            temp=(55.0, 60.0),
            elapsed=(3600.0, 7200.0),
            joints=["middle"],
        ),
        _envelope(
            "mem_hot_thumb_jnr",
            body="rh56_right_01",
            regime_label="THERMAL_TRACKING_DEGRADATION",
            envelope_type="validated",
            temp=(55.0, 60.0),
            elapsed=(3600.0, 7200.0),
            joints=["thumb"],
        ),
        _envelope(
            "mem_cold_middle_jnr",
            body="rh56_right_01",
            regime_label="COLD_HEALTHY",
            envelope_type="observed",
            temp=(30.0, 52.0),
            elapsed=(0.0, 1800.0),
            joints=["middle"],
            confidence=0.6,
            evidence_count=1,
            success_count=0,
        ),
        _envelope(
            "mem_cold2_index_jnr",
            body="rh56_right_01",
            regime_label="COLD_HEALTHY",
            envelope_type="observed",
            temp=(30.0, 52.0),
            joints=["index"],
            confidence=0.6,
            evidence_count=1,
            success_count=0,
        ),
        _envelope(
            "mem_left_middle_jnr",
            body="rh56_left_01",
            regime_label="COLD_HEALTHY",
            envelope_type="observed",
            temp=(30.0, 52.0),
            joints=["middle"],
            confidence=0.6,
            evidence_count=1,
            success_count=0,
        ),
        # run1 death spiral: cooldown/slowdown memory CONTRAINDICATED in
        # healthy regimes (PR #98 run1 patch proofs).
        _envelope(
            "mem_hot_middle_jnr",
            body="rh56_right_01",
            regime_label="COLD_HEALTHY",
            envelope_type="contraindicated",
            temp=(30.0, 52.0),
            joints=["middle"],
            reason="breaks_reveal_timing",
            confidence=0.9,
            evidence_count=3,
            success_count=0,
        ),
        _envelope(
            "mem_hot_middle_jnr",
            body="rh56_right_01",
            regime_label="WARM_STABLE",
            envelope_type="contraindicated",
            temp=(45.0, 54.0),
            joints=["middle"],
            reason="breaks_reveal_timing",
            confidence=0.9,
            evidence_count=3,
            success_count=0,
        ),
    ]
    return memories, envelopes


def queries() -> list[dict[str, Any]]:
    """Machine-anchored regime queries (ground truth by construction).

    Each query names its regime context; the corpus session the memory came
    from decides truth — a hot memory in a cold query is
    ``semantically_related_but_inapplicable``.
    """
    return [
        {
            "query_id": "q_cold_middle_jnr",
            "text": "右手 middle 未达到目标位置 joint_not_reached 怎么办",
            "regime": "cold",
            "body_id": "rh56_right_01",
            "joint_name": "middle",
            "relevant": ["mem_cold_middle_jnr", "mem_hot_middle_jnr"],
            "applicable": ["mem_cold_middle_jnr"],
            "applicable_validated": [],
            "semantically_related_but_inapplicable": ["mem_hot_middle_jnr"],
            "contraindicated": ["mem_hot_middle_jnr"],
        },
        {
            "query_id": "q_warm_middle_jnr",
            "text": "middle joint_not_reached again, short session",
            "regime": "warm",
            "body_id": "rh56_right_01",
            "joint_name": "middle",
            "relevant": ["mem_cold_middle_jnr", "mem_hot_middle_jnr"],
            "applicable": [],
            "applicable_validated": [],
            "semantically_related_but_inapplicable": ["mem_hot_middle_jnr"],
            "contraindicated": ["mem_hot_middle_jnr"],
        },
        {
            "query_id": "q_hot_middle_jnr",
            "text": "57度两小时会话 middle 又 joint_not_reached",
            "regime": "hot",
            "body_id": "rh56_right_01",
            "joint_name": "middle",
            "relevant": ["mem_hot_middle_jnr", "mem_cold_middle_jnr"],
            "applicable": ["mem_hot_middle_jnr"],
            "applicable_validated": ["mem_hot_middle_jnr"],
            "semantically_related_but_inapplicable": ["mem_cold_middle_jnr"],
            "contraindicated": [],
        },
        {
            "query_id": "q_cold_left_middle",
            "text": "左手 middle 未到位",
            "regime": "cold",
            "body_id": "rh56_left_01",
            "joint_name": "middle",
            "relevant": ["mem_left_middle_jnr"],
            "applicable": ["mem_left_middle_jnr"],
            "applicable_validated": [],
            "semantically_related_but_inapplicable": [],
            "contraindicated": [],
        },
        {
            "query_id": "q_cold_d435i",
            "text": "RealSense 启动时卡死没有图像",
            "regime": "cold",
            "body_id": None,
            "joint_name": None,
            "relevant": ["mem_d435i_wedge"],
            "applicable": [],
            "applicable_validated": [],
            "semantically_related_but_inapplicable": [],
            "contraindicated": [],
        },
    ]
