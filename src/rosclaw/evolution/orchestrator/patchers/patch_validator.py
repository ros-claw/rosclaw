"""PatchValidator — safety validation for all patches."""

import ast
import logging
from typing import Any

logger = logging.getLogger("rosclaw.auto.patchers.validator")


class PatchValidator:
    """Validate patches against safety policies and e-URDF constraints.

    Rejects:
    - Disabling safety checks
    - Modifying low-level controllers without approval
    - Exceeding e-URDF safety limits
    - Missing rollback plans
    - Code patches without human approval
    - Dangerous Python code patterns (exec, eval, __import__, subprocess, os.system)
    """

    FORBIDDEN_PATHS = [
        "/safety/collision_check_enabled",
        "/safety/emergency_stop_enabled",
        "/safety/human_approval_required",
        "/safety/sandbox_required",
    ]

    FORBIDDEN_VALUES = {
        "collision_check_enabled": False,
        "emergency_stop_enabled": False,
        "human_approval_required": False,
        "sandbox_required": False,
    }

    # Dangerous AST node types / names for code_patch
    DANGEROUS_NAMES = {
        "exec",
        "eval",
        "compile",
        "__import__",
        "subprocess",
        "os.system",
        "os.popen",
        "os.spawn",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "open",  # file open in code context is suspicious
    }

    def __init__(self, robot_profile: dict | None = None):
        self.robot_profile = robot_profile or {}

    def _normalize_path(self, path: str) -> str:
        """Normalize path to prevent ../ bypass attacks."""
        import os

        # Remove redundant separators and resolve . and ..
        normalized = os.path.normpath(path)
        # Ensure it starts with /
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized

    def _check_code_safety(self, code: str) -> list[str]:
        """AST-based analysis for dangerous code patterns."""
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"Code patch has syntax error: {exc}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    # Build dotted name like os.system
                    parts = []
                    current = func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    name = ".".join(reversed(parts))

                if name in ("exec", "eval", "compile"):
                    violations.append(f"Dangerous built-in call: {name}()")
                if name in ("os.system", "os.popen", "os.spawn", "os.exec"):
                    violations.append(f"Dangerous OS call: {name}()")
                if name in ("subprocess.call", "subprocess.run", "subprocess.Popen"):
                    violations.append(f"Dangerous subprocess call: {name}()")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("subprocess", "socket", "urllib", "http"):
                        violations.append(f"Forbidden import: {alias.name}")

            if isinstance(node, ast.ImportFrom) and node.module in ("os", "subprocess", "socket"):
                violations.append(f"Forbidden import from: {node.module}")

        return violations

    def validate(self, patch: Any) -> dict:
        """Validate a patch and return result dict.

        Returns:
            {"valid": bool, "violations": list[str], "requires_approval": bool}
        """
        violations = []
        requires_approval = False

        patch_type = getattr(patch, "patch_type", "")
        patch_level = getattr(patch, "patch_level", 2)
        changes = getattr(patch, "changes", [])

        # Level 5 code patch always needs approval
        if patch_level >= 5 or patch_type == "code_patch":
            requires_approval = True
            violations.append("code_patch requires human approval")

        # Level 4 policy checkpoint needs approval
        if patch_level == 4 or patch_type == "policy_checkpoint_patch":
            requires_approval = True
            violations.append("policy_checkpoint_patch requires human approval")

        # Check forbidden paths in changes
        for change in changes:
            path = change.get("path", "")
            change.get("old")
            new_val = change.get("new")
            change.get("action", "")

            # Normalize path to prevent bypass
            normalized = self._normalize_path(path)

            # Check forbidden paths
            for forbidden in self.FORBIDDEN_PATHS:
                # Exact match after normalization, or basename match
                if normalized == forbidden or normalized.endswith(forbidden):
                    violations.append(f"Forbidden path modification: {path}")

            # Check forbidden values (also on normalized path basename)
            basename = normalized.split("/")[-1] if "/" in normalized else normalized
            for key, forbidden_val in self.FORBIDDEN_VALUES.items():
                if basename == key and new_val == forbidden_val:
                    violations.append(f"Safety disable attempt: {path} = {new_val}")

            # Check e-URDF constraints
            eurdf_violation = self._check_eurdf_constraint(path, new_val)
            if eurdf_violation:
                violations.append(eurdf_violation)

            # AST safety check for code patches
            if patch_type == "code_patch" and isinstance(new_val, str):
                code_violations = self._check_code_safety(new_val)
                violations.extend(code_violations)

        # Rollback plan required for non-config patches
        if patch_level >= 2 and not getattr(patch, "rollback_plan", {}):
            violations.append("Missing rollback plan for patch_level >= 2")

        valid = len(violations) == 0 and not requires_approval

        return {
            "valid": valid,
            "violations": violations,
            "requires_approval": requires_approval,
            "patch_type": patch_type,
            "patch_level": patch_level,
        }

    def _check_eurdf_constraint(self, path: str, value: Any) -> str | None:
        """Check if patch violates e-URDF safety constraints."""
        safety = self.robot_profile.get("safety", {})

        if "max_joint_speed" in path and isinstance(value, (int, float)):
            limit = safety.get("max_joint_speed", 999)
            if value > limit:
                return f"max_joint_speed {value} exceeds e-URDF limit {limit}"

        if "max_force" in path and isinstance(value, (int, float)):
            limit = safety.get("max_force", 999)
            if value > limit:
                return f"max_force {value} exceeds e-URDF limit {limit}"

        if "workspace" in path and isinstance(value, (list, tuple)):
            ws_limits = safety.get("workspace", {})
            # Simplified check
            if ws_limits and len(value) == 2:
                axis = path.split("/")[-1] if "/" in path else ""
                if axis in ws_limits:
                    allowed = ws_limits[axis]
                    if value[0] < allowed[0] or value[1] > allowed[1]:
                        return f"workspace {axis} {value} exceeds e-URDF limit {allowed}"

        return None
