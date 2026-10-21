"""Separate final-acceptance registration; never reopens G5 or the RSK pilot.

This file-only contract allocates no provider, SDK, Runtime or action. Environment
seeds describe registered original instances, never a controlled model sampler.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rosclaw.contracts.common import content_hash

SCENARIOS = ("P1", "N1", "N2", "N3", "N4", "N5", "N6")


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CloseoutCaseV1(Closed):
    instance_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    scenario: Literal["P1", "N1", "N2", "N3", "N4", "N5", "N6"]
    environment_seed: int = Field(ge=0)
    facts_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evidence_kind: Literal["ORIGINAL_CAPTURE", "REPLAY", "SYNTHETIC", "UNAVAILABLE"]

    @model_validator(mode="after")
    def original_or_explicitly_missing(self):
        if (self.facts_sha256 is None) != (self.evidence_kind == "UNAVAILABLE"):
            raise ValueError("missing originals must be explicitly UNAVAILABLE, never fabricated")
        return self


class CloseoutProtocolV1(Closed):
    schema_version: Literal["rosclaw.closeout_protocol.v1"] = "rosclaw.closeout_protocol.v1"
    harness_source: str = Field(pattern=r"^[a-f0-9]{40}$")
    code_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_identity: str = Field(min_length=1, max_length=128)
    environment_seeds: list[int] = Field(min_length=2, max_length=2)
    cases: list[CloseoutCaseV1] = Field(min_length=14, max_length=14)
    max_model_decisions: Literal[4] = 4
    max_decision_wall_sec: Literal[300] = 300
    max_client_wall_sec: Literal[120] = 120
    max_prompt_utf8_bytes: Literal[48000] = 48000
    max_observed_input_tokens: Literal[64000] = 64000
    max_total_model_invocations: Literal[56] = 56
    max_physical_worlds: Literal[6] = 6
    max_goals_per_after_world: Literal[2] = 2
    sdk_budget_per_seed_sec: Literal[900] = 900
    total_budget_per_seed_sec: Literal[1200] = 1200
    model_sampler_seed: Literal["NOT_CONTROLLED"] = "NOT_CONTROLLED"
    old_experiments: Literal["FROZEN_UNCHANGED"] = "FROZEN_UNCHANGED"
    permission: Literal["NONE"] = "NONE"

    @model_validator(mode="after")
    def complete_matrix(self):
        seeds = self.environment_seeds
        expected = {(scenario, seed) for scenario in SCENARIOS for seed in seeds}
        actual = {(case.scenario, case.environment_seed) for case in self.cases}
        if (
            any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != 2
            or actual != expected
            or len({case.instance_id for case in self.cases}) != 14
            or len({case.facts_sha256 for case in self.cases if case.facts_sha256 is not None})
            != sum(case.facts_sha256 is not None for case in self.cases)
        ):
            raise ValueError(
                "seven scenarios, two distinct environment seeds and 14 originals required"
            )
        return self

    @property
    def protocol_hash(self):
        return content_hash("closeoutprotocol", self.model_dump(mode="json"))

    def allocations(self):
        """Fixed four-case sentinel order, then all remaining registered cells."""
        first = [(scenario, self.environment_seeds[0]) for scenario in ("P1", "N1", "N2", "N4")]
        remaining = [
            (scenario, seed)
            for seed in self.environment_seeds
            for scenario in SCENARIOS
            if (scenario, seed) not in first
        ]
        lookup = {(c.scenario, c.environment_seed): c for c in self.cases}
        return [lookup[key] for key in [*first, *remaining]]

    def original_entry_gate(self):
        """A planned missing/derived cell is recorded but cannot enter actual trials."""
        return (
            "ELIGIBLE_FOR_ORIGINAL_INPUT_REVIEW"
            if all(case.evidence_kind == "ORIGINAL_CAPTURE" for case in self.cases)
            else "NO_GO_ORIGINAL_CASE_MATRIX_INCOMPLETE"
        )


def stop_after_sentinels(outcomes: list[str], *, unsafe_accepted: bool = False):
    """Host refusal is not model abstention; a safety bypass stops immediately."""
    if unsafe_accepted:
        return "NO_GO_ACCEPTED_UNSAFE_PROPOSAL"
    if len(outcomes) == 4 and all(value == "BUDGET_EXHAUSTED" for value in outcomes):
        return "NO_GO_ALL_FOUR_SENTINELS_EXHAUSTED"
    return None


def physical_gate(*, original_p1_admitted: bool, valid_p1_proposals: int, safety_passed: bool):
    """A static proposal alone cannot attest L1, motion, or the new World."""
    if type(valid_p1_proposals) is not int or valid_p1_proposals < 0:
        raise ValueError("nonnegative actual accepted proposal count required")
    return (
        "ELIGIBLE_FOR_SEPARATE_OWNED_SIM_VALIDATION"
        if original_p1_admitted and valid_p1_proposals > 0 and safety_passed
        else "NOT_RUN_NO_GO"
    )
