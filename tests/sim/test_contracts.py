"""仿真契约骨架测试（PR-MH0，ADR-0014，红→绿）。"""

from __future__ import annotations

import re

import pytest

from rosclaw.contracts.common import UnsupportedVersionError, ValidationError
from rosclaw.sim import contracts as sc

ALL_CONTRACTS = [
    sc.SimulationBackendCapabilities,
    sc.ModelReference,
    sc.ModelInspection,
    sc.StateSnapshot,
    sc.ObservationRequest,
    sc.ObservationResult,
    sc.RolloutRequest,
    sc.SimulationTrace,
    sc.ModelPatch,
    sc.ModelPatchResult,
    sc.AuditRequest,
    sc.AuditResult,
    sc.ExperimentBranch,
    sc.ExperimentResult,
    sc.ComparisonResult,
    sc.SimulationEvidenceBundle,
]

SCHEMA_RE = re.compile(r"^rosclaw\.sim\.[a-z0-9_]+\.v1$")

# 与 tests/architecture/test_invariants.py 保持同一词表（ADR-0000 §2）。
SECRET_FIELD_RE = re.compile(
    r"(api_key|secret|password|passwd|private_key|access_token|refresh_token"
    r"|bearer|permit_secret|hmac_key)",
    re.IGNORECASE,
)


@pytest.mark.parametrize("cls", ALL_CONTRACTS, ids=lambda c: c.__name__)
def test_schema_literal_v1(cls) -> None:
    assert SCHEMA_RE.match(cls.SCHEMA), cls.SCHEMA
    assert cls.supported_major() == 1
    assert cls().schema_version == cls.SCHEMA


@pytest.mark.parametrize("cls", ALL_CONTRACTS, ids=lambda c: c.__name__)
def test_envelope_fields_present(cls) -> None:
    obj = cls()
    assert obj.backend == ""
    assert obj.backend_version == ""
    assert obj.created_at == ""
    assert obj.digest == ""


@pytest.mark.parametrize("cls", ALL_CONTRACTS, ids=lambda c: c.__name__)
def test_no_secret_like_fields(cls) -> None:
    for name in cls.model_fields:
        assert not SECRET_FIELD_RE.search(name), f"{cls.__name__}.{name}"


def test_canonical_digest_stability() -> None:
    kwargs = {
        "backend": "mujoco",
        "backend_version": "3.11.0",
        "created_at": "2026-09-14T00:00:00+00:00",
        "model_ref": "simmdl_0123456789abcdef",
        "model_digest": "sha256:00",
    }
    a = sc.ModelReference(**kwargs)
    b = sc.ModelReference(**kwargs)
    assert a.canonical_hash() == b.canonical_hash()

    c = sc.ModelReference(**{**kwargs, "model_digest": "sha256:ff"})
    assert c.canonical_hash() != a.canonical_hash()


def test_digest_field_excluded_from_hash_and_idempotent() -> None:
    obj = sc.ModelReference(
        backend="mujoco", created_at="2026-09-14T00:00:00+00:00", model_ref="simmdl_0" * 1
    )
    stamped = obj.with_digest()
    assert stamped.digest.startswith("simref_")
    # digest 字段自身不参与哈希：盖章前后 canonical_hash 不变。
    assert stamped.canonical_hash() == obj.canonical_hash()
    # 幂等：重复盖章结果一致。
    assert stamped.with_digest().digest == stamped.digest


@pytest.mark.parametrize("cls", ALL_CONTRACTS, ids=lambda c: c.__name__)
def test_unknown_field_forward_compat(cls) -> None:
    payload = {"schema_version": cls.SCHEMA, "future_field": {"x": 1}}
    obj = cls.model_validate_contract(payload)
    assert obj.future_field == {"x": 1}


@pytest.mark.parametrize("cls", ALL_CONTRACTS, ids=lambda c: c.__name__)
def test_unknown_major_version_fail_closed(cls) -> None:
    stem = cls.SCHEMA.rsplit(".v", 1)[0]
    with pytest.raises(UnsupportedVersionError):
        cls.model_validate_contract({"schema_version": f"{stem}.v2"})


def test_wrong_stem_fail_closed() -> None:
    with pytest.raises(ValidationError):
        sc.ModelReference.model_validate_contract(
            {"schema_version": "rosclaw.sim.state_snapshot.v1"}
        )
    with pytest.raises(ValidationError):
        sc.ModelReference.model_validate_contract({"schema_version": "not-a-schema"})


def test_model_reference_lineage_fields() -> None:
    ref = sc.ModelReference(
        model_ref="simmdl_aaaaaaaaaaaaaaaa",
        parent_model_ref="simmdl_bbbbbbbbbbbbbbbb",
        source={"kind": "eurdf", "ref": "ur5e"},
        compiled=True,
    )
    assert ref.parent_model_ref == "simmdl_bbbbbbbbbbbbbbbb"
    assert ref.source["kind"] == "eurdf"
    assert ref.compiled is True


def test_evidence_bundle_never_usable_for_real() -> None:
    bundle = sc.SimulationEvidenceBundle()
    assert bundle.trust_level == "SIMULATED"
    assert bundle.usable_for_real_execution is False
