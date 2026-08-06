from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from rosclaw_know.contracts import PUBLIC_CONTRACTS as KNOW_PUBLIC_CONTRACTS

from rosclaw.knowledge.context_adapter import build_how_context
from rosclaw.knowledge.contracts import (
    PUBLIC_CONTRACTS,
    HowAdviceRequestV2,
    KnowledgeUsageFeedbackV1,
)


def test_public_contracts_are_versioned_and_forbid_unknown_fields():
    for contract in PUBLIC_CONTRACTS:
        assert contract.SCHEMA_VERSION
        assert contract.model_config["extra"] == "forbid"


def test_advice_request_round_trip_and_strict_version():
    request = HowAdviceRequestV2(
        request_id="request_1",
        mode="consult",
        query="Which implementation should be inspected?",
        context=build_how_context(task="inspection", robot_model="limo"),
    )
    assert HowAdviceRequestV2.validate_wire_json(request.to_wire_json()) == request
    with pytest.raises(ValidationError):
        HowAdviceRequestV2.model_validate(
            {**request.model_dump(mode="json"), "schema_version": "rosclaw.how.advice_request.v3"}
        )


def test_feedback_requires_aware_timestamp():
    with pytest.raises(ValidationError):
        KnowledgeUsageFeedbackV1(
            feedback_id="feedback_1",
            reference_pack_id="pack_1",
            knowledge_unit_id="unit_1",
            verdict="useful",
            context_hash="a" * 64,
            origin="user",
            created_at=datetime.now(),
        )
    feedback = KnowledgeUsageFeedbackV1(
        feedback_id="feedback_1",
        reference_pack_id="pack_1",
        knowledge_unit_id="unit_1",
        verdict="useful",
        context_hash="a" * 64,
        origin="user",
        created_at=datetime.now(UTC),
    )
    assert feedback.schema_version == "rosclaw.knowledge_usage_feedback.v1"


def test_core_wire_schema_matches_authoritative_know_package():
    authoritative = {
        contract.SCHEMA_VERSION: contract.model_json_schema(mode="serialization")
        for contract in KNOW_PUBLIC_CONTRACTS
    }
    for contract in PUBLIC_CONTRACTS:
        version = contract.SCHEMA_VERSION
        if version in authoritative:
            assert contract.model_json_schema(mode="serialization") == authoritative[version]
