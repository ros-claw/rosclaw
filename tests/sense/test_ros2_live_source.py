"""No ROS required: original-source age, unknown values and clock epochs."""

import math
from types import SimpleNamespace

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.sense.collectors.ros2_collector import ROS2Collector, _battery, _imu
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.runtime import SenseRuntime


def battery(stamp=10.0, **changes):
    values = {
        "header": SimpleNamespace(
            stamp=SimpleNamespace(sec=int(stamp), nanosec=int((stamp - int(stamp)) * 1e9))
        ),
        "percentage": 0.75,
        "voltage": 14.0,
        "current": -0.5,
        "power_supply_status": 2,
        "present": True,
    }
    return SimpleNamespace(**(values | changes))


def collector(monkeypatch, **topics):
    import rosclaw.sense.collectors.ros2_collector as module

    clock = {"ros": 10.0, "mono": 100.0, "wall": 1000.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["mono"])
    monkeypatch.setattr(module.time, "time", lambda: clock["wall"])
    c = ROS2Collector(
        "fixture", topics=topics or {"battery": {"topic": "/robot/battery", "max_age_s": 1.5}}
    )
    c._ros_now = lambda: clock["ros"]
    c._running = True
    return c, clock


def test_unconfigured_never_enrolls_guessed_topics_or_refreshes_data():
    c = ROS2Collector()
    c.start()
    state = c.collect()
    assert state.timestamp == 0 and state.energy.battery_percent is None
    assert state.source == "ros2:unconfigured" and not state.raw["collector_configured"]
    c.stop()


@pytest.mark.parametrize("topic", ["relative", "/", "//battery", "/robot/1battery", "/bad-name"])
def test_topic_bindings_are_explicit_valid_absolute_names(topic):
    with pytest.raises(ValueError):
        ROS2Collector(topics={"battery": {"topic": topic, "max_age_s": 1.5}})


@pytest.mark.parametrize("age", [0, -1, float("nan"), float("inf")])
def test_invalid_age_cannot_configure_a_healthy_source(age):
    with pytest.raises(ValueError):
        ROS2Collector(topics={"battery": {"topic": "/battery", "max_age_s": age}})


def test_typed_roles_and_clock_flag_do_not_silently_coerce():
    with pytest.raises(ValueError):
        ROS2Collector(topics={"arbitrary": {"topic": "/battery", "max_age_s": 1.0}})
    with pytest.raises(ValueError):
        ROS2Collector(use_sim_time="false")


def test_battery_uses_message_units_and_absence_unknown():
    e = _battery(battery())
    assert e.battery_percent == 75 and e.current_draw == 0.5 and e.voltage == 14
    assert _battery(battery(current=0.5, power_supply_status=1)).current_draw == -0.5
    assert _battery(battery(present=False)).battery_percent is None
    assert (
        _battery(battery(percentage=75.0, voltage=-1, current=float("nan"))).battery_percent is None
    )
    assert _battery(battery(percentage=float("nan"))).battery_percent is None


def test_repeated_collect_preserves_original_time_and_copies_values(monkeypatch):
    c, t = collector(monkeypatch)
    c._receive("battery", battery())
    first = c.collect()
    assert first.timestamp == 1000 and first.energy.battery_percent == 75
    first.energy.battery_percent = 0
    t.update(wall=1001, mono=101, ros=11)
    later = c.collect()
    assert later.timestamp == 1000 and later.energy.battery_percent == 75
    assert later.raw["roles"]["battery"]["original_header_stamp_s"] == 10
    assert not later.raw["action_authority"] and not later.raw["body_identity_verified"]


@pytest.mark.parametrize("age_source", ["ros", "mono"])
def test_either_original_clock_age_or_receive_age_can_make_values_unknown(monkeypatch, age_source):
    c, t = collector(monkeypatch)
    c._receive("battery", battery())
    t[age_source] += 2
    state = c.collect()
    assert state.energy.battery_percent is None
    assert state.timestamp == 1000 and state.raw["roles"]["battery"]["fresh"] is False


def test_replayed_header_cannot_renew_receive_freshness(monkeypatch):
    c, t = collector(monkeypatch)
    c._receive("battery", battery())
    t.update(ros=11, mono=101, wall=1001)
    c._receive("battery", battery(percentage=0.99))
    assert c.collect().energy.battery_percent == 75
    t.update(ros=11.4, mono=102, wall=1002)
    assert c.collect().energy.battery_percent is None
    assert c.collect().timestamp == 1000


@pytest.mark.parametrize("stamp", [0, 8.0, 12.0])
def test_missing_stale_future_header_never_populates_body(monkeypatch, stamp):
    c, _ = collector(monkeypatch)
    c._receive("battery", battery(stamp))
    assert c.collect().energy.battery_percent is None and c.collect().timestamp == 0


def test_clock_reset_invalidates_prior_epoch_and_requires_new_source_data(monkeypatch):
    c, t = collector(monkeypatch)
    c._receive("battery", battery())
    epoch = c.collect().raw["observation_epoch"]
    t["ros"] = 1.0
    assert c.collect().energy.battery_percent is None
    assert c.collect().raw["observation_epoch"] == epoch + 1
    c._receive("battery", battery(1.0, percentage=0.5))
    assert c.collect().energy.battery_percent == 50


def test_stop_invalidates_values_and_late_callbacks_do_not_resurrect_them(monkeypatch):
    c, t = collector(monkeypatch)
    c._receive("battery", battery())
    c.stop()
    t["ros"] = 11
    c._receive("battery", battery(11.0))
    assert c.collect().energy.battery_percent is None
    assert not c.collect().raw["transport_running"]


def test_partial_roles_do_not_invent_other_healthy_channels(monkeypatch):
    c, _ = collector(
        monkeypatch,
        battery={"topic": "/battery", "max_age_s": 1.5},
        imu={"topic": "/imu", "max_age_s": 1.5},
    )
    c._receive("battery", battery())
    state = c.collect()
    assert state.energy.battery_percent == 75 and state.imu.pitch_deg is None
    assert state.raw["roles"]["imu"]["fresh"] is False
    assert state.communication.heartbeat_ok is None and state.perception.status == "unknown"


def test_imu_unavailable_covariance_and_invalid_quaternion_are_unknown():
    def msg(q=None, cov=0):
        return SimpleNamespace(
            orientation=q or SimpleNamespace(x=0.0, y=0.0, z=math.sqrt(0.5), w=math.sqrt(0.5)),
            orientation_covariance=[cov] * 9,
            angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=1.0),
            angular_velocity_covariance=[cov] * 9,
        )

    assert _imu(msg()).yaw_deg == pytest.approx(90.0)
    assert _imu(msg(cov=-1)).yaw_deg is None and _imu(msg(cov=-1)).angular_velocity is None
    assert _imu(msg(q=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0))).yaw_deg is None


def test_sense_factory_passes_only_explicit_operator_telemetry_config():
    runtime = SenseRuntime(
        config=SenseConfig(
            robot_id="fixture",
            collector="ros2",
            extra={
                "ros2_topics": {"battery": {"topic": "/battery", "max_age_s": 3.0}},
                "ros2_use_sim_time": True,
            },
        ),
        event_bus=EventBus(),
    )
    assert runtime._collector.bindings["battery"].topic == "/battery"
    assert runtime._collector.use_sim_time is True
    assert runtime._collector.collect().energy.battery_percent is None


def test_transport_failure_invalidates_prior_values(monkeypatch):
    c, _ = collector(monkeypatch)
    c._receive("battery", battery())

    def failed():
        raise RuntimeError("fixture transport ended")

    c._executor = SimpleNamespace(spin=failed)
    c._spin()
    state = c.collect()
    assert state.energy.battery_percent is None and not state.raw["transport_running"]
    assert state.raw["error"] == "RuntimeError"


def test_owned_context_cannot_be_overwritten_without_cleanup(monkeypatch):
    c, _ = collector(monkeypatch)
    c._running = False
    owned = object()
    c._context = owned
    with pytest.raises(RuntimeError, match="owns a context"):
        c.start()
    assert c._context is owned


def test_unstopped_thread_refuses_restart_and_earns_no_values(monkeypatch):
    c, _ = collector(monkeypatch)
    c._receive("battery", battery())
    owned = object()
    c._context = owned
    c._thread = SimpleNamespace(join=lambda **kw: None, is_alive=lambda: True)
    c._executor = SimpleNamespace(shutdown=lambda **kw: False)
    with pytest.raises(RuntimeError, match="did not stop"):
        c.stop()
    assert c.collect().energy.battery_percent is None
    with pytest.raises(RuntimeError, match="owns a context"):
        c.start()
    assert c._context is owned


def test_bindings_do_not_mutate_after_collector_construction():
    config = {"battery": {"topic": "/battery", "max_age_s": 1.0}}
    c = ROS2Collector(topics=config)
    config["battery"]["max_age_s"] = 1000
    assert c.bindings["battery"].max_age_s == 1
    with pytest.raises(TypeError):
        c.bindings["battery"] = None


@pytest.mark.parametrize("source", ["ros2:unconfigured", "ros2:unavailable", "sense:degraded"])
def test_unavailable_transports_are_not_presented_as_unverified_live_data(source):
    from rosclaw.sense.explain import _observation_kind

    assert _observation_kind(source) == "unavailable"
    assert _observation_kind("ros2:live") == "unverified"
