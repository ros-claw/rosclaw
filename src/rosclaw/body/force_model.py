"""Generic force model for physical feedback frames.

Manages per-DOF force baselines and classifies net-force magnitudes into
contact levels. The units are body-defined (grams-force for RH56, Newtons
for larger arms, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ForceBaseline:
    """Per-DOF baseline statistics."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    samples: int = 0


@dataclass
class DofForceWindow:
    """Per-DOF force thresholds."""

    desired_min: float = 0.0
    desired_max: float = 0.0
    hard: float = 0.0
    emergency: float = 0.0


class ForceModel:
    """Body-agnostic force baseline and contact-level classifier."""

    def __init__(
        self,
        baseline: Optional[Dict[str, ForceBaseline]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        contact_windows: Optional[Dict[str, DofForceWindow]] = None,
        policy: Optional[Dict[str, bool]] = None,
    ):
        self.baseline: Dict[str, ForceBaseline] = baseline or {}
        self.thresholds: Dict[str, float] = thresholds or self._default_thresholds()
        self.contact_windows: Dict[str, DofForceWindow] = contact_windows or {}
        self.policy: Dict[str, bool] = policy or self._default_policy()

    @staticmethod
    def _default_thresholds() -> Dict[str, float]:
        return {
            "soft_contact": 0.0,
            "contact_detect_sigma": 5.0,
        }

    @staticmethod
    def _default_policy() -> Dict[str, bool]:
        return {
            "baseline_required": True,
            "use_force_for_static_contact": True,
            "use_current_for_static_contact": False,
        }

    def window(self, dof: str) -> DofForceWindow:
        return self.contact_windows.get(dof, DofForceWindow())

    def net_force(
        self,
        force_raw: Dict[str, Optional[float]],
        dofs: Optional[List[str]] = None,
    ) -> Dict[str, Optional[float]]:
        """Subtract baseline mean from raw force for each DOF."""
        dofs = dofs or list(force_raw.keys())
        result: Dict[str, Optional[float]] = {}
        for name in dofs:
            raw = force_raw.get(name)
            if raw is None:
                result[name] = None
                continue
            baseline = self.baseline.get(name, ForceBaseline()).mean
            result[name] = raw - baseline
        return result

    def contact_level(self, net_force: Optional[float], dof: str) -> Optional[str]:
        """Classify a single net-force magnitude into a contact level."""
        if net_force is None:
            return None
        w = self.window(dof)
        if w.emergency and net_force >= w.emergency:
            return "emergency"
        if w.hard and net_force >= w.hard:
            return "hard"
        if w.desired_max and net_force >= w.desired_max:
            return "strong"
        if w.desired_min and net_force >= w.desired_min:
            return "desired"
        soft = self.thresholds.get("soft_contact", 0.0)
        if soft and net_force >= soft:
            return "soft"
        return "none"

    def is_contact(
        self,
        net_force: Optional[float],
        dof: str,
        min_force: Optional[float] = None,
    ) -> bool:
        if net_force is None:
            return False
        threshold = min_force if min_force is not None else self.window(dof).desired_min
        return threshold is not None and net_force >= threshold

    def is_desired_contact(self, net_force: Optional[float], dof: str) -> bool:
        if net_force is None:
            return False
        w = self.window(dof)
        if w.desired_min is None or w.desired_max is None:
            return False
        return w.desired_min <= net_force <= w.desired_max

    def is_over_contact(self, net_force: Optional[float], dof: str) -> bool:
        if net_force is None:
            return False
        hard = self.window(dof).hard
        return hard is not None and net_force >= hard

    def list_missing_baselines(self, dofs: List[str]) -> List[str]:
        """Return DOFs with no baseline samples."""
        return [
            name
            for name in dofs
            if self.baseline.get(name, ForceBaseline()).samples == 0
        ]

    def save_policy_yaml(self, path: Path) -> None:
        """Write force policy config to YAML."""
        data = {
            "force_policy": {
                "baseline_required": self.policy.get("baseline_required", True),
                "use_force_for_static_contact": self.policy.get(
                    "use_force_for_static_contact", True
                ),
                "use_current_for_static_contact": self.policy.get(
                    "use_current_for_static_contact", False
                ),
                "contact_windows": {
                    name: {
                        "desired_min": w.desired_min,
                        "desired_max": w.desired_max,
                        "hard": w.hard,
                        "emergency": w.emergency,
                    }
                    for name, w in self.contact_windows.items()
                },
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
