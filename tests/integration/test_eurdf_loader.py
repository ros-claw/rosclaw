"""Tests for e-URDF Loader — Physical DNA Registry."""

import pytest

from rosclaw.runtime import (
    EURDFLoader,
    RobotCapabilityProfile,
    RobotEmbodimentProfile,
    RobotRegistry,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)


class TestEURDFLoader:
    """Test EURDFLoader loading and validation."""

    def test_list_robots_finds_ur5e(self):
        loader = EURDFLoader()
        robots = loader.list_robots()
        assert "ur5e" in robots

    def test_load_ur5e_complete_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")

        assert profile.robot_id == "universal_robots_ur5e"
        assert profile.name == "UR5e"
        assert profile.vendor == "Universal Robots"
        assert profile.version == "5.12"
        assert profile.embodiment.dof == 6

    def test_embodiment_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        emb = profile.embodiment

        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 9
        assert len(emb.joints) == 6
        assert len(emb.sensors) == 4
        assert len(emb.actuators) == 6
        assert emb.joints[0]["name"] == "shoulder_pan_joint"

    def test_safety_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        safety = profile.safety

        assert isinstance(safety, RobotSafetyProfile)
        assert safety.safety_level == "STRICT"
        assert "pfl" in safety.to_dict()
        assert safety.pfl.get("max_tcp_force") == 150.0

    def test_capability_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        cap = profile.capability

        assert isinstance(cap, RobotCapabilityProfile)
        assert len(cap.capabilities) == 5
        cap_names = [c["name"] for c in cap.capabilities]
        assert "pick_and_place" in cap_names
        assert "force_compliant_insert" in cap_names

    def test_simulation_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        sim = profile.simulation

        assert isinstance(sim, RobotSimulationProfile)
        assert "mujoco" in sim.backends
        assert "isaac" in sim.backends
        assert "gazebo" in sim.backends

    def test_semantic_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        sem = profile.semantic

        assert isinstance(sem, RobotSemanticProfile)
        assert "collaborative_robot" in sem.semantic_tags
        assert len(sem.functional_regions) > 0

    def test_benchmark_profile(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        bench = profile.benchmark

        assert bench.robot_id == "universal_robots_ur5e"
        assert len(bench.kinematic_benchmarks) > 0
        assert len(bench.task_benchmarks) > 0

    def test_validate_ur5e_passes(self):
        loader = EURDFLoader()
        result = loader.validate("ur5e")

        assert result["valid"] is True
        assert "robot.eurdf.yaml" in result["files_found"]
        assert len(result["errors"]) == 0

    def test_validate_missing_robot_fails(self):
        loader = EURDFLoader()
        result = loader.validate("nonexistent_robot")

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_load_missing_robot_raises(self):
        loader = EURDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_robot")

    def test_to_dict_roundtrip(self):
        loader = EURDFLoader()
        profile = loader.load("ur5e")
        d = profile.to_dict()

        assert d["robot_id"] == "universal_robots_ur5e"
        assert "embodiment" in d
        assert "safety" in d
        assert "capability" in d
        assert "simulation" in d
        assert "semantic" in d
        assert "benchmark" in d


class TestRobotRegistry:
    """Test RobotRegistry caching and lookup."""

    def test_list_available_includes_ur5e(self):
        reg = RobotRegistry()
        available = reg.list_available()
        assert "ur5e" in available

    def test_install_and_get(self):
        reg = RobotRegistry()
        profile = reg.install("ur5e")
        assert profile.robot_id == "universal_robots_ur5e"

        # Get by directory name
        cached = reg.get("ur5e")
        assert cached is not None
        assert cached.robot_id == "universal_robots_ur5e"

        # Get by canonical robot_id
        cached2 = reg.get("universal_robots_ur5e")
        assert cached2 is not None

    def test_list_registered(self):
        reg = RobotRegistry()
        reg.install("ur5e")
        registered = reg.list()
        assert "ur5e" in registered
        assert "universal_robots_ur5e" in registered

    def test_get_auto_install(self):
        reg = RobotRegistry()
        profile = reg.get("ur5e")
        assert profile is not None
        assert profile.name == "UR5e"

    def test_get_nonexistent_returns_none(self):
        reg = RobotRegistry()
        assert reg.get("nonexistent_robot") is None

    def test_inspect_returns_dict(self):
        reg = RobotRegistry()
        reg.install("ur5e")
        d = reg.inspect("ur5e")
        assert d["robot_id"] == "universal_robots_ur5e"
        assert "embodiment" in d

    def test_inspect_missing_raises(self):
        reg = RobotRegistry()
        with pytest.raises(FileNotFoundError):
            reg.inspect("nonexistent_robot")
