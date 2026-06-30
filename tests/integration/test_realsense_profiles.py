"""Tests for RealSense D400-series sensor profiles.

Covers the three builtin Python profiles:
- realsense_d405
- realsense_d435i
- realsense_dual
"""

from rosclaw.eurdf_zoo.profiles import (
    REALSENSE_D405_PROFILE,
    REALSENSE_D435I_PROFILE,
    REALSENSE_DUAL_PROFILE,
)
from rosclaw.runtime import (
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotEmbodimentProfile,
    RobotRegistry,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)


class TestRealSenseD405:
    """Test Intel RealSense D405 profile."""

    def test_registry_lists_d405(self):
        reg = RobotRegistry()
        available = reg.list_available()
        assert "realsense_d405" in available
        assert "realsense-d405" in available

    def test_load_complete_profile(self):
        reg = RobotRegistry()
        profile = reg.get("realsense-d405")
        assert profile is not None
        assert profile.robot_id == "realsense_d405"
        assert profile.name == "Intel RealSense D405"
        assert profile.vendor == "Intel RealSense"
        assert profile.version == "1.0"
        assert profile.embodiment.dof == 0

    def test_embodiment_profile(self):
        profile = REALSENSE_D405_PROFILE
        emb = profile.embodiment
        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 11
        assert len(emb.joints) == 10
        assert len(emb.sensors) == 4
        assert len(emb.actuators) == 0
        sensor_types = {s["type"] for s in emb.sensors}
        assert sensor_types == {"camera", "depth_camera"}

    def test_safety_profile(self):
        profile = REALSENSE_D405_PROFILE
        safety = profile.safety
        assert isinstance(safety, RobotSafetyProfile)
        assert safety.safety_level == "STRICT"
        assert safety.safety_limits.get("perception_only") is True
        assert safety.environment.get("real_robot_execution_allowed") is False
        assert safety.environment.get("sandbox_required") is True

    def test_capability_profile(self):
        profile = REALSENSE_D405_PROFILE
        cap = profile.capability
        assert isinstance(cap, RobotCapabilityProfile)
        assert len(cap.capabilities) == 3
        cap_ids = {c["id"] for c in cap.capabilities}
        assert "rgb_camera" in cap_ids
        assert "depth_camera" in cap_ids
        assert "stereo_infrared" in cap_ids
        cap_names = {c["name"] for c in cap.capabilities}
        assert "rgb_observation" in cap_names
        assert "depth_observation" in cap_names
        forbidden = cap.skill_registry.get("forbidden_capabilities", [])
        assert len(forbidden) == 2

    def test_simulation_profile(self):
        profile = REALSENSE_D405_PROFILE
        sim = profile.simulation
        assert isinstance(sim, RobotSimulationProfile)
        assert "mujoco" in sim.backends
        assert "ros2" in sim.backends
        assert "rviz" in sim.backends

    def test_semantic_profile(self):
        profile = REALSENSE_D405_PROFILE
        sem = profile.semantic
        assert isinstance(sem, RobotSemanticProfile)
        assert "rgbd_camera" in sem.semantic_tags
        assert "depth_camera" in sem.semantic_tags
        assert len(sem.functional_regions) == 4
        assert len(sem.visual_features) == 2

    def test_benchmark_profile(self):
        profile = REALSENSE_D405_PROFILE
        bench = profile.benchmark
        assert isinstance(bench, RobotBenchmarkProfile)
        assert bench.robot_id == "realsense_d405"


class TestRealSenseD435i:
    """Test Intel RealSense D435i profile."""

    def test_registry_lists_d435i(self):
        reg = RobotRegistry()
        available = reg.list_available()
        assert "realsense_d435i" in available
        assert "realsense-d435i" in available

    def test_load_complete_profile(self):
        reg = RobotRegistry()
        profile = reg.get("realsense-d435i")
        assert profile is not None
        assert profile.robot_id == "realsense_d435i"
        assert profile.name == "Intel RealSense D435i"
        assert profile.embodiment.dof == 0

    def test_embodiment_profile(self):
        profile = REALSENSE_D435I_PROFILE
        emb = profile.embodiment
        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 15
        assert len(emb.joints) == 14
        assert len(emb.sensors) == 5
        assert len(emb.actuators) == 0
        sensor_types = {s["type"] for s in emb.sensors}
        assert "imu" in sensor_types

    def test_semantic_profile(self):
        profile = REALSENSE_D435I_PROFILE
        sem = profile.semantic
        assert isinstance(sem, RobotSemanticProfile)
        assert "imu" in sem.semantic_tags
        assert len(sem.functional_regions) == 5

    def test_safety_blocks_real_execution(self):
        profile = REALSENSE_D435I_PROFILE
        assert profile.safety.environment.get("real_robot_execution_allowed") is False


class TestRealSenseDual:
    """Test dual RealSense composite profile."""

    def test_registry_lists_dual(self):
        reg = RobotRegistry()
        available = reg.list_available()
        assert "realsense_dual" in available
        assert "realsense-dual" in available

    def test_load_complete_profile(self):
        reg = RobotRegistry()
        profile = reg.get("realsense-dual")
        assert profile is not None
        assert profile.robot_id == "realsense_dual"
        assert profile.name == "Dual RealSense D405 + D435i"

    def test_embodiment_profile(self):
        profile = REALSENSE_DUAL_PROFILE
        emb = profile.embodiment
        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 13
        assert len(emb.joints) == 12
        assert len(emb.sensors) == 5
        assert len(emb.actuators) == 0
        sensor_names = {s["name"] for s in emb.sensors}
        assert "head_color_camera" in sensor_names
        assert "wrist_imu" in sensor_names

    def test_semantic_profile(self):
        profile = REALSENSE_DUAL_PROFILE
        sem = profile.semantic
        assert isinstance(sem, RobotSemanticProfile)
        assert "dual_camera" in sem.semantic_tags
        assert len(sem.functional_regions) == 3


class TestRegistry:
    """Test RobotRegistry with RealSense profiles."""

    def test_install_and_get_d405(self):
        reg = RobotRegistry()
        profile = reg.install("realsense-d405")
        assert profile.robot_id == "realsense_d405"
        assert reg.get("realsense_d405") is not None
        assert reg.get("realsense-d405") is not None

    def test_install_and_get_d435i(self):
        reg = RobotRegistry()
        profile = reg.install("realsense-d435i")
        assert profile.robot_id == "realsense_d435i"
        assert reg.get("realsense_d435i") is not None

    def test_install_and_get_dual(self):
        reg = RobotRegistry()
        profile = reg.install("realsense-dual")
        assert profile.robot_id == "realsense_dual"
        assert reg.get("realsense_dual") is not None

    def test_inspect_all(self):
        reg = RobotRegistry()
        for rid in ["realsense-d405", "realsense-d435i", "realsense-dual"]:
            reg.install(rid)
            d = reg.inspect(rid)
            assert "embodiment" in d
            assert "safety" in d
            assert "capability" in d
            assert "simulation" in d
            assert "semantic" in d
            assert "benchmark" in d
