"""Native admission tests for the bounded supplied-record adapters."""

import math
import struct
from types import SimpleNamespace

import pytest

from rosclaw.perception import cloud_quality, scan_quality
from rosclaw.perception.supplied_observations import (
    analyze_supplied_cloud,
    analyze_supplied_scan,
)


def _valid_scan_payload():
    return {
        "record": {
            "header": {"stamp": {"sec": -1, "nanosec": 0}, "frame_id": "controlled"},
            "angle_min": 0.0,
            "angle_max": 0.25,
            "angle_increment": 0.25,
            "time_increment": 0.0,
            "scan_time": 0.0,
            "range_min": 0.1,
            "range_max": 4.0,
            "ranges": [1.0, 2.0],
            "intensities": [],
        },
        "reference_time_ns": -1_000_000_000,
    }


def _valid_cloud_payload():
    data = struct.pack("<fff", 0.5, 0.25, 1.0)
    return {
        "record": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "controlled"},
            "width": 1,
            "height": 1,
            "point_step": 12,
            "row_step": 12,
            "is_bigendian": False,
            "is_dense": True,
            "fields": [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            ],
            "data": list(data),
        }
    }


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    if value == "NAN":
        return math.nan
    if value == "POS_INF":
        return math.inf
    if value == "NEG_INF":
        return -math.inf
    return value


def test_scan_adapter_matches_canonical_analyzer():
    payload = _valid_scan_payload()
    expected = scan_quality.analyze_scan(_ns(payload["record"]), payload["reference_time_ns"])
    assert analyze_supplied_scan(payload) == expected
    assert expected["status"] == "CLEAR"


def test_scan_nonfinite_tags_reach_canonical_quality_rules():
    payload = _valid_scan_payload()
    payload["record"]["ranges"] = ["NAN", "POS_INF"]
    result = analyze_supplied_scan(payload)
    assert result["status"] == "NO_VALID_RETURN"
    assert result["invalid_count"] == 2


def test_scan_signed_stamp_and_negative_reference_admitted():
    payload = _valid_scan_payload()
    payload["record"]["header"]["stamp"]["sec"] = -2_000_000_000
    payload["reference_time_ns"] = -1_999_999_000_000_000_000
    result = analyze_supplied_scan(payload)
    assert result["status"] == "STALE"


def test_cloud_adapter_matches_canonical_analyzer():
    payload = _valid_cloud_payload()
    message = _ns(payload["record"])
    message.data = bytes(payload["record"]["data"])
    expected = cloud_quality.analyze_cloud(message)
    assert analyze_supplied_cloud(payload) == expected
    assert expected["status"] == "VALID_CLOUD"


def test_cloud_declared_dimension_17_is_quality_invalid_not_transport_error():
    payload = _valid_cloud_payload()
    payload["record"]["width"] = 17
    result = analyze_supplied_cloud(payload)
    assert result["status"] == "QUALITY_INVALID"
    assert result["total_count"] == 17
    assert result["invalid_count"] == 17


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update({"extra": 0}),
        lambda p: p.update({"reference_time_ns": True}),
        lambda p: p.update({"reference_time_ns": 1 << 64}),
        lambda p: p["record"].update({"extra": 0}),
        lambda p: p["record"].update({"ranges": [1.0] * 4097}),
        lambda p: p["record"].update({"intensities": [1.0] * 4097}),
        lambda p: p["record"].update({"ranges": [True]}),
        lambda p: p["record"].update({"ranges": [math.nan]}),
        lambda p: p["record"].update({"ranges": ["INF"]}),
        lambda p: p["record"]["header"]["stamp"].update({"sec": True}),
        lambda p: p["record"]["header"]["stamp"].update({"sec": 1 << 31}),
        lambda p: p["record"]["header"]["stamp"].update({"nanosec": 1_000_000_000}),
        lambda p: p["record"]["header"].update({"frame_id": "x" * 129}),
        lambda p: p["record"].update({"angle_min": 2e12}),
    ],
)
def test_scan_malformed_rejected_before_conversion(mutate):
    payload = _valid_scan_payload()
    mutate(payload)
    with pytest.raises((TypeError, ValueError)):
        analyze_supplied_scan(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update({"extra": 0}),
        lambda p: p["record"].update({"width": True}),
        lambda p: p["record"].update({"width": 65536}),
        lambda p: p["record"].update({"row_step": 1 << 32}),
        lambda p: p["record"].update({"is_bigendian": 0}),
        lambda p: p["record"].update({"fields": p["record"]["fields"] * 11}),
        lambda p: p["record"].update({"data": [0] * 65537}),
        lambda p: p["record"].update({"data": [256]}),
        lambda p: p["record"].update({"data": [True]}),
        lambda p: p["record"]["fields"][0].update({"name": "x" * 65}),
        lambda p: p["record"]["fields"][0].update({"datatype": 256}),
    ],
)
def test_cloud_malformed_rejected_before_conversion(mutate):
    payload = _valid_cloud_payload()
    mutate(payload)
    with pytest.raises((TypeError, ValueError)):
        analyze_supplied_cloud(payload)


def test_subclass_containers_and_scalars_rejected():
    class DerivedDict(dict):
        pass

    class DerivedList(list):
        pass

    class DerivedInt(int):
        pass

    class DerivedStr(str):
        pass

    with pytest.raises(TypeError):
        analyze_supplied_scan(DerivedDict(_valid_scan_payload()))
    payload = _valid_scan_payload()
    payload["record"]["ranges"] = DerivedList(payload["record"]["ranges"])
    with pytest.raises(TypeError):
        analyze_supplied_scan(payload)
    payload = _valid_scan_payload()
    payload["reference_time_ns"] = DerivedInt(payload["reference_time_ns"])
    with pytest.raises(TypeError):
        analyze_supplied_scan(payload)
    payload = _valid_scan_payload()
    payload["record"]["header"]["frame_id"] = DerivedStr("controlled")
    with pytest.raises(TypeError):
        analyze_supplied_scan(payload)
    cloud = _valid_cloud_payload()
    cloud["record"]["data"] = DerivedList(cloud["record"]["data"])
    with pytest.raises(TypeError):
        analyze_supplied_cloud(cloud)


def test_user_iteration_and_bytes_hooks_never_executed():
    class Trap:
        def __iter__(self):
            raise AssertionError("USER_ITERATOR_EXECUTED")

        def __bytes__(self):
            raise AssertionError("USER_BYTES_EXECUTED")

        def __getattr__(self, name):
            raise AssertionError("USER_PROPERTY_EXECUTED")

    payload = _valid_scan_payload()
    payload["record"]["ranges"] = Trap()
    with pytest.raises((TypeError, ValueError)):
        analyze_supplied_scan(payload)
    cloud = _valid_cloud_payload()
    cloud["record"]["data"] = Trap()
    with pytest.raises((TypeError, ValueError)):
        analyze_supplied_cloud(cloud)


def _long_named(cls, name):
    return type(name, (cls,), {})


def _long_type_name_payloads():
    """One malformed payload per direct type-rejection branch, with long names."""
    names = [
        "A" * 1024,
        "B" * 100000,
        "类型" * 512,
        "センサー" * 20000,
    ]
    cases = []
    for name in names:
        # dict branch (root payload)
        cases.append(("scan", _long_named(dict, name)(_valid_scan_payload()), name))
        # list branch (record.ranges)
        scan = _valid_scan_payload()
        scan["record"]["ranges"] = _long_named(list, name)(scan["record"]["ranges"])
        cases.append(("scan", scan, name))
        # int branch (reference_time_ns)
        scan = _valid_scan_payload()
        scan["reference_time_ns"] = _long_named(int, name)(scan["reference_time_ns"])
        cases.append(("scan", scan, name))
        # bool branch (cloud record.is_dense)
        cloud = _valid_cloud_payload()
        cloud["record"]["is_dense"] = _long_named(int, name)(1)
        cases.append(("cloud", cloud, name))
        # str branch (header.frame_id)
        scan = _valid_scan_payload()
        scan["record"]["header"]["frame_id"] = _long_named(str, name)("controlled")
        cases.append(("scan", scan, name))
        # scalar branch (record.angle_min gets a non-number object of long-named type)
        scan = _valid_scan_payload()
        scan["record"]["angle_min"] = _long_named(object, name)()
        cases.append(("scan", scan, name))
    return cases


@pytest.mark.parametrize("kind,payload,name", _long_type_name_payloads())
def test_long_type_names_rejected_with_bounded_diagnostic(kind, payload, name):
    analyzer = {"scan": analyze_supplied_scan, "cloud": analyze_supplied_cloud}[kind]
    with pytest.raises(TypeError) as excinfo:
        analyzer(payload)
    message = str(excinfo.value)
    # Diagnostic stays bounded and never echoes the caller's class name.
    assert len(message.encode()) <= 512
    assert name[:64] not in message
    assert "must be a plain" in message


def test_long_type_names_rejected_before_canonical_body(monkeypatch):
    scan_calls = []
    cloud_calls = []
    monkeypatch.setattr(scan_quality, "analyze_scan", lambda *a, **k: scan_calls.append(1))
    monkeypatch.setattr(cloud_quality, "analyze_cloud", lambda *a, **k: cloud_calls.append(1))
    for kind, payload, _name in _long_type_name_payloads():
        analyzer = {"scan": analyze_supplied_scan, "cloud": analyze_supplied_cloud}[kind]
        with pytest.raises(TypeError):
            analyzer(payload)
    assert scan_calls == [] and cloud_calls == []


def test_input_payload_not_mutated():
    payload = _valid_scan_payload()
    snapshot = repr(sorted(payload["record"].items())) + repr(payload["reference_time_ns"])
    analyze_supplied_scan(payload)
    assert repr(sorted(payload["record"].items())) + repr(payload["reference_time_ns"]) == snapshot


def _hardening_cases():
    cloud = _valid_cloud_payload()
    many_keys_scan = _valid_scan_payload()
    many_keys_scan.update({f"unexpected_{n}": 0 for n in range(10000)})
    many_keys_cloud = _valid_cloud_payload()
    many_keys_cloud.update({f"unexpected_{n}": 0 for n in range(10000)})
    long_key = _valid_scan_payload()
    long_key["UNTRUSTED_KEY_" + ("x" * 100000)] = 0
    long_tag = _valid_scan_payload()
    long_tag["record"]["angle_min"] = "UNTRUSTED_TAG_" + ("y" * 100000)
    nonstring_key = _valid_scan_payload()
    nonstring_key[123] = "unused"
    cases = [
        ("scan", many_keys_scan, "unexpected_"),
        ("cloud", many_keys_cloud, "unexpected_"),
        ("scan", long_key, "UNTRUSTED_KEY_"),
        ("cloud", {**cloud, "UNTRUSTED_KEY_" + ("x" * 100000): 0}, "UNTRUSTED_KEY_"),
        ("scan", long_tag, "UNTRUSTED_TAG_"),
        ("scan", nonstring_key, ""),
    ]
    for bits in (4096, 100000):
        huge_scalar = _valid_scan_payload()
        huge_scalar["record"]["angle_min"] = 1 << bits
        cases.append(("scan", huge_scalar, ""))
        huge_bound = _valid_scan_payload()
        huge_bound["reference_time_ns"] = 1 << bits
        cases.append(("scan", huge_bound, ""))
    return cases


@pytest.mark.parametrize("kind,payload,marker", _hardening_cases())
def test_hardening_bounded_typed_rejection_without_echo(kind, payload, marker):
    analyzer = {"scan": analyze_supplied_scan, "cloud": analyze_supplied_cloud}[kind]
    with pytest.raises((TypeError, ValueError)) as excinfo:
        analyzer(payload)
    message = str(excinfo.value)
    assert len(message.encode()) <= 512
    assert not marker or marker not in message


def test_hardening_rejections_happen_before_canonical_body(monkeypatch):
    scan_calls = []
    cloud_calls = []
    original_scan = scan_quality.analyze_scan
    original_cloud = cloud_quality.analyze_cloud

    def spy_scan(*args, **kwargs):
        scan_calls.append(1)
        return original_scan(*args, **kwargs)

    def spy_cloud(*args, **kwargs):
        cloud_calls.append(1)
        return original_cloud(*args, **kwargs)

    monkeypatch.setattr(scan_quality, "analyze_scan", spy_scan)
    monkeypatch.setattr(cloud_quality, "analyze_cloud", spy_cloud)
    for kind, payload, _marker in _hardening_cases():
        analyzer = {"scan": analyze_supplied_scan, "cloud": analyze_supplied_cloud}[kind]
        with pytest.raises((TypeError, ValueError)):
            analyzer(payload)
    assert scan_calls == [] and cloud_calls == []


def test_catalog_descriptors_publish_exact_input_schemas():
    import jsonschema

    from rosclaw.agentd.tooling.catalog import ToolCatalog
    from rosclaw.perception.capabilities import register_perception_tools

    catalog = ToolCatalog()
    register_perception_tools(catalog)
    scan_schema = catalog.get("perception.scan_quality").input_schema
    cloud_schema = catalog.get("perception.cloud_quality").input_schema
    jsonschema.Draft7Validator.check_schema(scan_schema)
    jsonschema.Draft7Validator.check_schema(cloud_schema)
    jsonschema.validate(_valid_scan_payload(), scan_schema)
    jsonschema.validate(_valid_cloud_payload(), cloud_schema)
    tagged = _valid_scan_payload()
    tagged["record"]["ranges"] = ["NAN", "POS_INF", "NEG_INF"]
    jsonschema.validate(tagged, scan_schema)
    wide = _valid_cloud_payload()
    wide["record"]["width"] = 17
    jsonschema.validate(wide, cloud_schema)
    for kind, payload, _marker in _hardening_cases():
        if kind != "scan":
            continue
        # Huge plain integers beyond the int64 reference bound are not JSON
        # transport values; the adapter (not the descriptor schema) owns them.
        reference = payload.get("reference_time_ns", 0)
        if type(reference) is int and abs(reference) > (1 << 63):
            continue
        if type(payload["record"]["angle_min"]) is int:
            continue
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, scan_schema)
