"""TeamMemberCardV1 (总纲 §10.3).

Discovery produces candidates only; formal join requires identity
authentication, body/capability verification, TeamPolicy acceptance and an
epoch commit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class MemberBody(ContractModel):
    SCHEMA = "rosclaw.member_body.v1"

    schema_version: Literal["rosclaw.member_body.v1"] = "rosclaw.member_body.v1"
    body_id: str
    effective_body_hash: str
    class_: str = Field(default="unknown", alias="class")


class MemberEndpoints(ContractModel):
    SCHEMA = "rosclaw.member_endpoints.v1"

    schema_version: Literal["rosclaw.member_endpoints.v1"] = "rosclaw.member_endpoints.v1"
    task: str | None = Field(None, description="e.g. a2a:https://blue-02.local/a2a")
    world: str | None = Field(None, description="e.g. zenoh:team/blue/member/02")


class MemberSecurity(ContractModel):
    SCHEMA = "rosclaw.member_security.v1"

    schema_version: Literal["rosclaw.member_security.v1"] = "rosclaw.member_security.v1"
    identity_key_id: str | None = None
    cert_fingerprint: str | None = None


class MemberHealth(ContractModel):
    SCHEMA = "rosclaw.member_health.v1"

    schema_version: Literal["rosclaw.member_health.v1"] = "rosclaw.member_health.v1"
    state: Literal["CANDIDATE", "JOINING", "READY", "SUSPECT", "LOST", "LEFT"] = "CANDIDATE"
    observed_at: str | None = None
    ttl_ms: int = 1000


class TeamMemberCardV1(ContractModel):
    SCHEMA = "rosclaw.team_member_card.v1"
    HASH_PREFIX = "tmcard"

    schema_version: Literal["rosclaw.team_member_card.v1"] = "rosclaw.team_member_card.v1"
    team_id: str
    member_id: str
    agent_card_url: str | None = None
    body: MemberBody
    capabilities: list[str] = Field(default_factory=list)
    endpoints: MemberEndpoints = Field(default_factory=MemberEndpoints)
    security: MemberSecurity = Field(default_factory=MemberSecurity)
    health: MemberHealth = Field(default_factory=MemberHealth)
    team_epoch: int = 0
