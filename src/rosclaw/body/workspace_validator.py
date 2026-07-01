"""Body validator — runs all checks and produces a BodyValidationReport."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from rosclaw.body.compiler import canonical_json, compute_checksum
from rosclaw.body.renderer import EmbodimentRenderer
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.safety import SafetyInvariantEngine
from rosclaw.body.schema import BodyValidationReport, EurdfProfile, ValidationResult
from rosclaw.runtime.eurdf_loader import RobotRegistry


class BodyValidator:
    """Validate body workspace files, schema, safety, and generated artifacts."""

    def __init__(self, resolver: BodyResolver | None = None):
        self.resolver = resolver or BodyResolver()
        self.safety_engine = SafetyInvariantEngine()

    def validate_all(self) -> BodyValidationReport:
        """Run full validation suite and return aggregated report."""
        checks: list[ValidationResult] = []

        # 0. Body linked
        if not self.resolver.is_linked():
            checks.append(
                ValidationResult(
                    check_id="body-linked",
                    status="block",
                    message="No body linked. Run: rosclaw body link-eurdf <profile_id>",
                    category="schema",
                )
            )
            return BodyValidationReport(result="BLOCKED", checks=checks, summary={"block": 1})

        # 1. body.yaml loads
        body = self._load_body(checks)

        # 2. e-URDF profile loads and checksum matches (best effort)
        _ = self._load_eurdf(body, checks)

        # 2a. Detect stale cached profile relative to current e-URDF source.
        self._check_eurdf_source_stale(body, checks)

        # 3. calibration.yaml loads
        calibration = self._load_calibration(checks)

        # 4. maintenance.log is valid JSONL
        maintenance = self._load_maintenance(checks)

        # 5. EMBODIMENT.md exists and has required markers
        self._validate_embodiment(checks)

        # 6. BODY.md alias exists
        self._validate_body_md_alias(checks)

        # 7. Safety invariants
        self._validate_safety_invariants(body, calibration, maintenance, checks)

        # 8. Generated summaries consistency
        self._validate_generated_summaries(body, checks)

        # 9. ID consistency
        self._validate_id_consistency(body, checks)

        summary = self._summarize(checks)
        result = self._result_from_summary(summary)
        return BodyValidationReport(result=result, checks=checks, summary=summary)

    def _load_body(self, checks: list[ValidationResult]) -> Any:
        try:
            body = self.resolver.get_current_body_yaml()
            checks.append(
                ValidationResult(
                    check_id="body-yaml-loads",
                    status="pass",
                    message="body.yaml loads successfully.",
                    category="schema",
                )
            )
            return body
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="body-yaml-loads",
                    status="fail",
                    message=f"body.yaml failed to load: {exc}",
                    category="schema",
                )
            )
            return None

    def _load_eurdf(self, body: Any, checks: list[ValidationResult]) -> Any:
        try:
            eurdf = self.resolver.get_current_eurdf_profile()
            expected_checksum = ""
            if body is not None:
                expected_checksum = body.model_ref.get("profile_checksum", "")
            if expected_checksum and self.resolver.eurdf_profile_path.exists():
                actual = compute_checksum(self.resolver.eurdf_profile_path)
                if actual != expected_checksum:
                    checks.append(
                        ValidationResult(
                            check_id="eurdf-checksum",
                            status="warn",
                            message=f"e-URDF checksum mismatch: expected {expected_checksum}, got {actual}.",
                            category="schema",
                        )
                    )
                else:
                    checks.append(
                        ValidationResult(
                            check_id="eurdf-checksum",
                            status="pass",
                            message="e-URDF checksum matches.",
                            category="schema",
                        )
                    )
            checks.append(
                ValidationResult(
                    check_id="eurdf-loads",
                    status="pass",
                    message="e-URDF profile loads successfully.",
                    category="schema",
                )
            )
            return eurdf
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="eurdf-loads",
                    status="fail",
                    message=f"e-URDF profile failed to load: {exc}",
                    category="schema",
                )
            )
            return None

    def _eurdf_fingerprint(self, eurdf: EurdfProfile) -> str:
        """Return a stable content fingerprint for an e-URDF profile."""
        return f"sha256:{hashlib.sha256(canonical_json(asdict(eurdf)).encode('utf-8')).hexdigest()}"

    def _check_eurdf_source_stale(self, body: Any, checks: list[ValidationResult]) -> None:
        """Compare cached e-URDF profile against the current source profile."""
        if body is None:
            return
        profile_id = body.model_ref.get("profile_id")
        if not profile_id:
            return
        if not self.resolver.eurdf_profile_path.exists():
            return

        try:
            cached = EurdfProfile.from_yaml(self.resolver.eurdf_profile_path)
            cached_fp = self._eurdf_fingerprint(cached)
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="eurdf-source-stale",
                    status="warn",
                    message=f"Could not fingerprint cached e-URDF profile: {exc}",
                    category="schema",
                )
            )
            return

        try:
            source_profile = RobotRegistry().get(profile_id)
            if source_profile is None:
                return
            source_eurdf = EurdfProfile.from_robot_complete_profile(source_profile)
            source_fp = self._eurdf_fingerprint(source_eurdf)
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="eurdf-source-stale",
                    status="warn",
                    message=f"Could not load source e-URDF profile '{profile_id}': {exc}",
                    category="schema",
                )
            )
            return

        if cached_fp != source_fp:
            checks.append(
                ValidationResult(
                    check_id="eurdf-source-stale",
                    status="warn",
                    message=(
                        f"Cached e-URDF profile is stale: source profile '{profile_id}' "
                        "has changed since the body was initialized. "
                        "Run 'rosclaw body render' or 'rosclaw body init --force' to refresh."
                    ),
                    category="schema",
                )
            )
        else:
            checks.append(
                ValidationResult(
                    check_id="eurdf-source-stale",
                    status="pass",
                    message="Cached e-URDF profile matches the current source profile.",
                    category="schema",
                )
            )

    def _load_calibration(self, checks: list[ValidationResult]) -> Any:
        try:
            calibration = self.resolver.get_calibration()
            status = calibration.overall_status()
            if status in ("valid", "validated"):
                checks.append(
                    ValidationResult(
                        check_id="calibration-status",
                        status="pass",
                        message=f"Calibration status is '{status}'.",
                        category="calibration",
                    )
                )
            else:
                checks.append(
                    ValidationResult(
                        check_id="calibration-status",
                        status="warn",
                        message=f"Calibration status is '{status}'; precision capabilities may be degraded.",
                        category="calibration",
                    )
                )
            return calibration
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="calibration-loads",
                    status="fail",
                    message=f"calibration.yaml failed to load: {exc}",
                    category="calibration",
                )
            )
            return None

    def _load_maintenance(self, checks: list[ValidationResult]) -> list[Any]:
        try:
            events = self.resolver.get_maintenance_events()
            checks.append(
                ValidationResult(
                    check_id="maintenance-log-loads",
                    status="pass",
                    message=f"maintenance.log contains {len(events)} event(s).",
                    category="maintenance",
                )
            )
            return events
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="maintenance-log-loads",
                    status="fail",
                    message=f"maintenance.log failed to load: {exc}",
                    category="maintenance",
                )
            )
            return []

    def _validate_embodiment(self, checks: list[ValidationResult]) -> None:
        path = self.resolver.embodiment_md_path
        if not path.exists():
            checks.append(
                ValidationResult(
                    check_id="embodiment-exists",
                    status="warn",
                    message="EMBODIMENT.md does not exist. Run: rosclaw body render",
                    category="render",
                )
            )
            return
        content = path.read_text(encoding="utf-8")
        required_markers = [
            EmbodimentRenderer.GENERATED_START,
            EmbodimentRenderer.GENERATED_END,
        ]
        missing = [m for m in required_markers if m not in content]
        if missing:
            checks.append(
                ValidationResult(
                    check_id="embodiment-markers",
                    status="fail",
                    message=f"EMBODIMENT.md missing markers: {missing}.",
                    category="render",
                )
            )
        else:
            checks.append(
                ValidationResult(
                    check_id="embodiment-markers",
                    status="pass",
                    message="EMBODIMENT.md has generated markers.",
                    category="render",
                )
            )
        required_sections = [
            "## 1. Identity",
            "## 6. Forbidden Capabilities",
            "## 7. Safety Limits",
            "## 8. Known Faults",
            "## 14. Machine-readable Summary",
        ]
        missing_sections = [s for s in required_sections if s not in content]
        if missing_sections:
            checks.append(
                ValidationResult(
                    check_id="embodiment-sections",
                    status="warn",
                    message=f"EMBODIMENT.md missing sections: {missing_sections}.",
                    category="render",
                )
            )
        else:
            checks.append(
                ValidationResult(
                    check_id="embodiment-sections",
                    status="pass",
                    message="EMBODIMENT.md contains required sections.",
                    category="render",
                )
            )

    def _validate_body_md_alias(self, checks: list[ValidationResult]) -> None:
        path = self.resolver.body_md_path
        if not path.exists():
            checks.append(
                ValidationResult(
                    check_id="body-md-alias",
                    status="warn",
                    message="BODY.md alias does not exist. Run: rosclaw body render",
                    category="render",
                )
            )
            return
        content = path.read_text(encoding="utf-8")
        if "EMBODIMENT.md" in content:
            checks.append(
                ValidationResult(
                    check_id="body-md-alias",
                    status="pass",
                    message="BODY.md alias points to EMBODIMENT.md.",
                    category="render",
                )
            )
        else:
            checks.append(
                ValidationResult(
                    check_id="body-md-alias",
                    status="warn",
                    message="BODY.md exists but does not reference EMBODIMENT.md.",
                    category="render",
                )
            )

    def _validate_safety_invariants(
        self,
        body: Any,
        calibration: Any,
        maintenance: list[Any],
        checks: list[ValidationResult],
    ) -> None:
        if body is None:
            return
        try:
            base_caps = body.get_capabilities_spec()
            mods = self.safety_engine.apply(body, maintenance, calibration, base_caps)
            if mods["disabled"] or mods["degraded"]:
                checks.append(
                    ValidationResult(
                        check_id="safety-invariants",
                        status="warn",
                        message="Safety invariants modified capabilities: "
                        f"disabled={mods['disabled']}, degraded={mods['degraded']}.",
                        category="safety",
                    )
                )
            else:
                checks.append(
                    ValidationResult(
                        check_id="safety-invariants",
                        status="pass",
                        message="No safety invariant modifications required.",
                        category="safety",
                    )
                )
            for warning in mods.get("warnings", []):
                checks.append(
                    ValidationResult(
                        check_id="safety-invariant-warning",
                        status="warn",
                        message=warning,
                        category="safety",
                    )
                )
        except Exception as exc:
            checks.append(
                ValidationResult(
                    check_id="safety-invariants",
                    status="fail",
                    message=f"Safety invariant check failed: {exc}",
                    category="safety",
                )
            )

    def _validate_generated_summaries(self, body: Any, checks: list[ValidationResult]) -> None:
        generated_dir = self.resolver.generated_dir
        required = ["body.summary.json", "embodiment.agent.json", "safety.summary.json"]
        missing = [name for name in required if not (generated_dir / name).exists()]
        if missing:
            checks.append(
                ValidationResult(
                    check_id="generated-summaries",
                    status="warn",
                    message=f"Generated summaries missing: {missing}. Run: rosclaw body render",
                    category="render",
                )
            )
        else:
            checks.append(
                ValidationResult(
                    check_id="generated-summaries",
                    status="pass",
                    message="Generated summaries exist.",
                    category="render",
                )
            )

    def _validate_id_consistency(self, body: Any, checks: list[ValidationResult]) -> None:
        if body is None:
            return
        instance_id = body.body_instance.get("id")
        identity_id = body.get_identity().get("robot_instance_id")
        if instance_id and identity_id and instance_id != identity_id:
            checks.append(
                ValidationResult(
                    check_id="id-consistency",
                    status="warn",
                    message=f"ID mismatch: body_instance.id={instance_id}, identity.robot_instance_id={identity_id}.",
                    category="schema",
                )
            )
        else:
            checks.append(
                ValidationResult(
                    check_id="id-consistency",
                    status="pass",
                    message="Body instance ID is consistent.",
                    category="schema",
                )
            )

    def _summarize(self, checks: list[ValidationResult]) -> dict[str, int]:
        summary: dict[str, int] = {}
        for check in checks:
            summary[check.status] = summary.get(check.status, 0) + 1
        return summary

    def _result_from_summary(self, summary: dict[str, int]) -> str:
        if summary.get("block", 0) > 0:
            return "BLOCKED"
        if summary.get("fail", 0) > 0:
            return "FAIL"
        if summary.get("warn", 0) > 0:
            return "PASS_WITH_WARNINGS"
        return "PASS"
