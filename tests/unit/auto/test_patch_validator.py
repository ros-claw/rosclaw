"""Tests for PatchValidator with AST and path normalization."""
from rosclaw.auto.core.patch import Patch
from rosclaw.auto.patchers.patch_validator import PatchValidator


class TestPatchValidator:
    """AUTO-PATCH-001~007: Patch safety validation."""

    def test_config_patch_valid(self):
        validator = PatchValidator()
        patch = Patch(
            id="p1", proposal_id="prop1", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/motion/max_velocity", "old": 0.5, "new": 0.8}],
        )
        result = validator.validate(patch)
        assert result["valid"] is True
        assert result["requires_approval"] is False

    def test_skill_parameter_patch_valid(self):
        validator = PatchValidator()
        patch = Patch(
            id="p2", proposal_id="prop2", patch_level=2,
            patch_type="skill_parameter_patch", target_skill="pick_v1",
            changes=[{"path": "/skill/gripper_force", "old": 10, "new": 15}],
            rollback_plan={"action": "revert"},
        )
        result = validator.validate(patch)
        assert result["valid"] is True

    def test_forbidden_safety_disable_rejected(self):
        validator = PatchValidator()
        patch = Patch(
            id="p3", proposal_id="prop3", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/safety/collision_check_enabled", "old": True, "new": False}],
        )
        result = validator.validate(patch)
        assert result["valid"] is False
        assert any("Safety disable" in v for v in result["violations"])

    def test_code_patch_requires_approval(self):
        validator = PatchValidator()
        patch = Patch(
            id="p4", proposal_id="prop4", patch_level=5,
            patch_type="code_patch", target_skill="pick_v1",
            changes=[{"path": "/controller.py", "old": "", "new": "def foo(): pass"}],
            human_approval_required=False,
        )
        result = validator.validate(patch)
        assert result["requires_approval"] is True
        assert result["valid"] is False

    def test_emergency_stop_disable_rejected(self):
        validator = PatchValidator()
        patch = Patch(
            id="p5", proposal_id="prop5", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/safety/emergency_stop_enabled", "old": True, "new": False}],
        )
        result = validator.validate(patch)
        assert result["valid"] is False
        assert any("Safety disable" in v for v in result["violations"])

    def test_eurdf_constraint_exceeded(self):
        validator = PatchValidator(robot_profile={"safety": {"max_joint_speed": 2.0}})
        patch = Patch(
            id="p6", proposal_id="prop6", patch_level=2,
            patch_type="skill_parameter_patch", target_skill="pick_v1",
            changes=[{"path": "/motion/max_joint_speed", "old": 1.0, "new": 5.0}],
            rollback_plan={"action": "revert"},
        )
        result = validator.validate(patch)
        assert result["valid"] is False
        assert any("exceeds e-URDF" in v for v in result["violations"])

    def test_path_traversal_bypass_blocked(self):
        """../ traversal should not bypass forbidden path detection."""
        validator = PatchValidator()
        patch = Patch(
            id="p7", proposal_id="prop7", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/foo/../safety/collision_check_enabled", "old": True, "new": False}],
        )
        result = validator.validate(patch)
        assert result["valid"] is False
        assert any("Forbidden path" in v or "Safety disable" in v for v in result["violations"])

    def test_ast_detects_exec_in_code_patch(self):
        validator = PatchValidator()
        patch = Patch(
            id="p8", proposal_id="prop8", patch_level=5,
            patch_type="code_patch", target_skill="pick_v1",
            changes=[{"path": "/evil.py", "old": "", "new": "exec('rm -rf /')"}],
            human_approval_required=True,
        )
        result = validator.validate(patch)
        assert any("exec" in v for v in result["violations"])

    def test_ast_detects_subprocess_in_code_patch(self):
        validator = PatchValidator()
        patch = Patch(
            id="p9", proposal_id="prop9", patch_level=5,
            patch_type="code_patch", target_skill="pick_v1",
            changes=[{"path": "/evil.py", "old": "", "new": "import subprocess\nsubprocess.run(['ls'])"}],
            human_approval_required=True,
        )
        result = validator.validate(patch)
        assert any("subprocess" in v for v in result["violations"])
