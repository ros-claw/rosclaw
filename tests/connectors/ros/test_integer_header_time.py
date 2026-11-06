"""Original integer clocks, inclusive +/-1ns boundaries and archive agreement."""

import gzip
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.finite_coverage_evidence import validate_archive
from rosclaw.connectors.ros.public_coverage.time_contract import (
    header_age_ok,
    integer_age_ok,
    stamp_ns,
)
from rosclaw.kernel import ActionState
from tests.connectors.ros.test_finite_coverage_evidence import fixture_run
from tests.connectors.ros.test_public_coverage_execution import setup_executor


@pytest.mark.parametrize(
    "age,ok",
    [
        (0, True),
        (-100_000_001, False),
        (-100_000_000, True),
        (-99_999_999, True),
        (1_499_999_999, True),
        (1_500_000_000, True),
        (1_500_000_001, False),
    ],
)
def test_exact_original_integer_interval_at_large_clock(age, ok):
    native = 2_186_200_000_000
    header = native - age
    assert integer_age_ok(native, header, 0.1, 1.5) is ok
    assert (
        header_age_ok(native / 1e9, header / 1e9, 0.1, 1.5, native_ns=native, header_ns=header)
        is ok
    )


@pytest.mark.parametrize("native_ns,header_ns", [(True, 1), (1, 1.0), (None, 1), (1, None)])
def test_partial_or_noninteger_facts_never_fall_back(native_ns, header_ns):
    with pytest.raises(ValueError):
        header_age_ok(1.0, 1.0, 0.1, 1.5, native_ns=native_ns, header_ns=header_ns)


def test_raw_projection_mismatch_and_noncanonical_stamp_are_rejected():
    with pytest.raises(ValueError, match="projection"):
        header_age_ok(
            2186.2, 2186.2, 0.1, 1.5, native_ns=2_186_200_000_000, header_ns=2_186_300_000_000
        )
    with pytest.raises(ValueError):
        stamp_ns(SimpleNamespace(sec=2186, nanosec=1_000_000_000))
    assert stamp_ns(SimpleNamespace(sec=2186, nanosec=300_000_000)) == 2_186_300_000_000
    assert header_age_ok(2186.2, 2186.3, 0.1, 1.5) is False  # explicitly legacy, missing originals


@pytest.mark.parametrize("native_form", ["division", "multiply", "stamp"])
@pytest.mark.parametrize("header_form", ["division", "multiply", "stamp"])
@pytest.mark.parametrize(
    "age,ok",
    [
        (0, True),
        (-100_000_001, False),
        (-100_000_000, True),
        (1_500_000_000, True),
        (1_500_000_001, False),
    ],
)
def test_legal_ros_seconds_projections_keep_the_same_exact_integer_age(
    native_form, header_form, age, ok
):
    native = 2_186_000_000_010
    header = native - age

    def seconds(value, form):
        sec, ns = divmod(value, 1_000_000_000)
        return {"division": value / 1e9, "multiply": value * 1e-9, "stamp": sec + ns * 1e-9}[form]

    assert (
        header_age_ok(
            seconds(native, native_form),
            seconds(header, header_form),
            0.1,
            1.5,
            native_ns=native,
            header_ns=header,
        )
        is ok
    )


@pytest.mark.parametrize(
    "age,ok",
    [
        (-100_000_001, False),
        (-100_000_000, True),
        (-99_999_999, True),
        (1_499_999_999, True),
        (1_500_000_000, True),
        (1_500_000_001, False),
    ],
)
@pytest.mark.parametrize("projection", ["canonical", "mixed_ros_producers"])
def test_actual_sdk_validator_uses_original_ns(age, ok, projection):
    _, executor, port, _ = setup_executor()
    native = 2_186_000_000_010
    header = native - age
    header_s = (
        header / 1e9
        if projection == "canonical"
        else header // 1_000_000_000 + (header % 1_000_000_000) * 1e-9
    )
    native_s = native / 1e9 if projection == "canonical" else native * 1e-9
    snapshot = port.snapshot()
    sources = {s.role: s for s in executor.binding.sources}
    facts = tuple(
        replace(
            f,
            header_native_s=None if sources[f.role].max_header_age_s is None else header_s,
            header_native_ns=None if sources[f.role].max_header_age_s is None else header,
        )
        for f in snapshot.sources
    )
    snapshot = replace(snapshot, native_s=native_s, native_ns=native, sources=facts)
    # Use odom/cleaner stop validation so the 1s TF contract remains separate.
    if ok:
        assert executor._validate(snapshot, work=False) is snapshot
    else:
        with pytest.raises(ValueError, match="freshness"):
            executor._validate(snapshot, work=False)


def test_archive_integer_boundary_and_legacy_contract_are_not_conflated(tmp_path):
    executor, _, action = fixture_run(tmp_path)
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED
    payload = json.loads(
        gzip.decompress(
            (tmp_path / result.verification_result["task_evidence"]["filename"]).read_bytes()
        )
    )
    sample = payload["final"]
    sample.update(native_s=2186.2, native_ns=2_186_200_000_000)
    for fact in sample["sources"]:
        if fact["header_native_s"] is not None:
            fact.update(header_native_s=2186.3, header_native_ns=2_186_300_000_000)
    # Only final is changed; native time remains monotonic and stop facts fresh.
    validate_archive(payload)
    for fact in sample["sources"]:
        if fact["role"] == "odom":
            fact.update(header_native_ns=2_186_300_000_001, header_native_s=2_186_300_000_001 / 1e9)
    with pytest.raises(ValueError, match="header freshness"):
        validate_archive(payload)
