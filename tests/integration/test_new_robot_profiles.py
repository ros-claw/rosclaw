"""Tests for new e-URDF-Zoo robot profiles: Franka Panda, Unitree Go2, Fetch."""


from rosclaw.runtime import (
    EURDFLoader,
    RobotRegistry,
    RobotEmbodimentProfile,
    RobotSafetyProfile,
    RobotCapabilityProfile,
    RobotSimulationProfile,
    RobotSemanticProfile,
    RobotBenchmarkProfile,
)
from rosclaw.eurdf_zoo.profiles import (
    FRANKA_PANDA_PROFILE,
    UNITREE_GO2_PROFILE,
    FETCH_ROBOT_PROFILE,
)


class TestFrankaPanda:
    """Test Franka Emika Panda profile."""

    def test_loader_finds_franka_panda(self):
        loader = EURDFLoader()
        robots = loader.list_robots()
        assert "franka_panda" in robots

    def test_load_complete_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        assert profile.robot_id == "franka_emika_panda"
        assert profile.name == "Panda"
        assert profile.vendor == "Franka Emika"
        assert profile.version == "4.0"
        assert profile.embodiment.dof == 7

    def test_embodiment_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        emb = profile.embodiment
        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 10
        assert len(emb.joints) == 7
        assert len(emb.sensors) == 4
        assert len(emb.actuators) == 7
        assert emb.joints[0]["name"] == "panda_joint1"

    def test_safety_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        safety = profile.safety
        assert isinstance(safety, RobotSafetyProfile)
        assert safety.safety_level == "STRICT"
        assert "pfl" in safety.to_dict()

    def test_capability_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        cap = profile.capability
        assert isinstance(cap, RobotCapabilityProfile)
        assert len(cap.capabilities) == 5
        cap_names = [c["name"] for c in cap.capabilities]
        assert "pick_and_place" in cap_names
        assert "insertion" in cap_names

    def test_simulation_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        sim = profile.simulation
        assert isinstance(sim, RobotSimulationProfile)
        assert "mujoco" in sim.backends
        assert "isaac" in sim.backends
        assert "gazebo" in sim.backends

    def test_semantic_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        sem = profile.semantic
        assert isinstance(sem, RobotSemanticProfile)
        assert "collaborative_robot" in sem.semantic_tags
        assert len(sem.functional_regions) > 0

    def test_benchmark_profile(self):
        loader = EURDFLoader()
        profile = loader.load("franka_panda")
        bench = profile.benchmark
        assert isinstance(bench, RobotBenchmarkProfile)
        assert bench.robot_id == "franka_emika_panda"
        assert len(bench.kinematic_benchmarks) > 0

    def test_validate_passes(self):
        loader = EURDFLoader()
        result = loader.validate("franka_panda")
        assert result["valid"] is True
        assert "robot.eurdf.yaml" in result["files_found"]
        assert len(result["errors"]) == 0

    def test_python_module_profile(self):
        profile = FRANKA_PANDA_PROFILE
        assert profile.robot_id == "franka_emika_panda"
        assert profile.embodiment.dof == 7
        assert profile.safety.safety_level == "STRICT"
        assert len(profile.capability.capabilities) == 5


class TestUnitreeGo2:
    """Test Unitree Go2 Quadruped profile."""

    def test_loader_finds_unitree_go2(self):
        loader = EURDFLoader()
        robots = loader.list_robots()
        assert "unitree_go2" in robots

    def test_load_complete_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        assert profile.robot_id == "unitree_go2"
        assert profile.name == "Go2"
        assert profile.vendor == "Unitree"
        assert profile.version == "1.1"
        assert profile.embodiment.dof == 12

    def test_embodiment_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        emb = profile.embodiment
        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 18
        assert len(emb.joints) == 12
        assert len(emb.sensors) == 5
        assert len(emb.actuators) == 12
        assert emb.joints[0]["name"] == "fr_hip_joint"

    def test_safety_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        safety = profile.safety
        assert isinstance(safety, RobotSafetyProfile)
        assert safety.safety_level == "MODERATE"
        assert "pfl" in safety.to_dict()

    def test_capability_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        cap = profile.capability
        assert isinstance(cap, RobotCapabilityProfile)
        assert len(cap.capabilities) == 5
        cap_names = [c["name"] for c in cap.capabilities]
        assert "walk" in cap_names
        assert "trot" in cap_names
        assert "stair_climb" in cap_names

    def test_simulation_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        sim = profile.simulation
        assert isinstance(sim, RobotSimulationProfile)
        assert "mujoco" in sim.backends
        assert "isaac" in sim.backends

    def test_semantic_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        sem = profile.semantic
        assert isinstance(sem, RobotSemanticProfile)
        assert "quadruped" in sem.semantic_tags
        assert len(sem.functional_regions) > 0

    def test_benchmark_profile(self):
        loader = EURDFLoader()
        profile = loader.load("unitree_go2")
        bench = profile.benchmark
        assert isinstance(bench, RobotBenchmarkProfile)
        assert bench.robot_id == "unitree_go2"
        assert len(bench.task_benchmarks) > 0

    def test_validate_passes(self):
        loader = EURDFLoader()
        result = loader.validate("unitree_go2")
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_python_module_profile(self):
        profile = UNITREE_GO2_PROFILE
        assert profile.robot_id == "unitree_go2"
        assert profile.embodiment.dof == 12
        assert profile.safety.safety_level == "MODERATE"
        assert len(profile.capability.capabilities) == 5


class TestFetchRobot:
    """Test Fetch Mobile Manipulator profile."""

    def test_loader_finds_fetch_robot(self):
        loader = EURDFLoader()
        robots = loader.list_robots()
        assert "fetch_robot" in robots

    def test_load_complete_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        assert profile.robot_id == "fetch_robot"
        assert profile.name == "Fetch"
        assert profile.vendor == "Fetch Robotics"
        assert profile.version == "1.0"
        assert profile.embodiment.dof == 8

    def test_embodiment_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        emb = profile.embodiment
        assert isinstance(emb, RobotEmbodimentProfile)
        assert len(emb.links) == 13
        assert len(emb.joints) == 8
        assert len(emb.sensors) == 4
        assert len(emb.actuators) == 8
        assert emb.joints[0]["name"] == "torso_lift_joint"

    def test_safety_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        safety = profile.safety
        assert isinstance(safety, RobotSafetyProfile)
        assert safety.safety_level == "MODERATE"
        assert "pfl" in safety.to_dict()

    def test_capability_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        cap = profile.capability
        assert isinstance(cap, RobotCapabilityProfile)
        assert len(cap.capabilities) == 5
        cap_names = [c["name"] for c in cap.capabilities]
        assert "navigate" in cap_names
        assert "pick_and_place" in cap_names
        assert "hand_over" in cap_names

    def test_simulation_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        sim = profile.simulation
        assert isinstance(sim, RobotSimulationProfile)
        assert "mujoco" in sim.backends
        assert "isaac" in sim.backends

    def test_semantic_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        sem = profile.semantic
        assert isinstance(sem, RobotSemanticProfile)
        assert "mobile_manipulator" in sem.semantic_tags
        assert len(sem.functional_regions) > 0

    def test_benchmark_profile(self):
        loader = EURDFLoader()
        profile = loader.load("fetch_robot")
        bench = profile.benchmark
        assert isinstance(bench, RobotBenchmarkProfile)
        assert bench.robot_id == "fetch_robot"
        assert len(bench.task_benchmarks) > 0

    def test_validate_passes(self):
        loader = EURDFLoader()
        result = loader.validate("fetch_robot")
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_python_module_profile(self):
        profile = FETCH_ROBOT_PROFILE
        assert profile.robot_id == "fetch_robot"
        assert profile.embodiment.dof == 8
        assert profile.safety.safety_level == "MODERATE"
        assert len(profile.capability.capabilities) == 5


class TestRegistry:
    """Test RobotRegistry with new profiles."""

    def test_list_available_includes_all(self):
        reg = RobotRegistry()
        available = reg.list_available()
        assert "franka_panda" in available
        assert "unitree_go2" in available
        assert "fetch_robot" in available

    def test_install_and_get_franka(self):
        reg = RobotRegistry()
        profile = reg.install("franka_panda")
        assert profile.robot_id == "franka_emika_panda"
        assert reg.get("franka_panda") is not None
        assert reg.get("franka_emika_panda") is not None

    def test_install_and_get_go2(self):
        reg = RobotRegistry()
        profile = reg.install("unitree_go2")
        assert profile.robot_id == "unitree_go2"
        assert reg.get("unitree_go2") is not None

    def test_install_and_get_fetch(self):
        reg = RobotRegistry()
        profile = reg.install("fetch_robot")
        assert profile.robot_id == "fetch_robot"
        assert reg.get("fetch_robot") is not None

    def test_inspect_all(self):
        reg = RobotRegistry()
        for rid in ["franka_panda", "unitree_go2", "fetch_robot"]:
            reg.install(rid)
            d = reg.inspect(rid)
            assert "embodiment" in d
            assert "safety" in d
            assert "capability" in d
            assert "simulation" in d
            assert "semantic" in d
            assert "benchmark" in d
