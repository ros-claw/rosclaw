"""L10: Safety policy tests with AST analysis."""
import pytest
from rosclaw.auto.patchers.patch_validator import PatchValidator
from rosclaw.auto.core.patch import Patch
from rosclaw.auto.promotion.gate import PromotionGate


class TestSafetyPolicy:
    """AUTO-SAFE-001~005: Safety must not be bypassable."""

    def test_real_runner_requires_human_approval(self):
        """AUTO-SAFE-001: Real robot runner must require human approval."""
        validator = PatchValidator()
        patch = Patch(
            id="p_real", proposal_id="prop_r", patch_level=3,
            patch_type="skill_graph_patch", target_skill="pick_v1",
            changes=[{"action": "real_robot_deploy"}],
            human_approval_required=True,
        )
        result = validator.validate(patch)
        # Level 3 requires human_optional approval
        assert result["requires_approval"] is True or result["valid"] is False

    def test_sandbox_cannot_be_skipped(self):
        """AUTO-SAFE-002: Sandbox required must be enforced."""
        validator = PatchValidator()
        patch = Patch(
            id="p_skip", proposal_id="prop_s", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/safety/sandbox_required", "old": True, "new": False}],
        )
        result = validator.validate(patch)
        assert result["valid"] is False
        assert any("Safety disable" in v for v in result["violations"])

    def test_collision_check_cannot_be_disabled(self):
        """AUTO-SAFE-003: collision_check cannot be disabled via patch."""
        validator = PatchValidator()
        patch = Patch(
            id="p_col", proposal_id="prop_c", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/safety/collision_check_enabled", "old": True, "new": False}],
        )
        result = validator.validate(patch)
        assert result["valid"] is False

    def test_path_traversal_bypass_blocked(self):
        """AUTO-SAFE-003b: ../ traversal cannot bypass forbidden path check."""
        validator = PatchValidator()
        patch = Patch(
            id="p_traversal", proposal_id="prop_t", patch_level=1,
            patch_type="config_patch", target_skill="pick_v1",
            changes=[{"path": "/config/../safety/collision_check_enabled", "old": True, "new": False}],
        )
        result = validator.validate(patch)
        assert result["valid"] is False
        assert any("Forbidden path" in v or "Safety disable" in v for v in result["violations"])

    def test_promotion_rejected_when_safety_regresses(self):
        """AUTO-SAFE-004: Success up but collision up = reject."""
        gate = PromotionGate()
        baseline = {"success_rate": 0.5, "collision_rate": 0.02}
        candidate = {"success_rate": 0.75, "collision_rate": 0.15}
        result = gate.evaluate(baseline, candidate)
        assert result.passed is False
        assert result.decision in ["reject", "need_more_data"]

    def test_no_approval_means_no_code_patch(self):
        """AUTO-SAFE-005: Code patch without approval is invalid."""
        validator = PatchValidator()
        patch = Patch(
            id="p_code", proposal_id="prop_code", patch_level=5,
            patch_type="code_patch", target_skill="pick_v1",
            changes=[{"path": "/controller.py", "old": "", "new": "def safe(): pass"}],
            human_approval_required=False,
        )
        result = validator.validate(patch)
        assert result["requires_approval"] is True
        assert result["valid"] is False

    def test_code_patch_with_exec_is_rejected(self):
        """AUTO-SAFE-006: Code patch containing exec() is rejected by AST."""
        validator = PatchValidator()
        patch = Patch(
            id="p_exec", proposal_id="prop_exec", patch_level=5,
            patch_type="code_patch", target_skill="pick_v1",
            changes=[{"path": "/controller.py", "old": "", "new": "exec('import os')"}],
            human_approval_required=True,
        )
        result = validator.validate(patch)
        assert any("exec" in v for v in result["violations"])

    def test_code_patch_with_subprocess_is_rejected(self):
        """AUTO-SAFE-007: Code patch containing subprocess is rejected by AST."""
        validator = PatchValidator()
        patch = Patch(
            id="p_subprocess", proposal_id="prop_sub", patch_level=5,
            patch_type="code_patch", target_skill="pick_v1",
            changes=[{"path": "/controller.py", "old": "", "new": "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"}],
            human_approval_required=True,
        )
        result = validator.validate(patch)
        assert any("subprocess" in v for v in result["violations"])
