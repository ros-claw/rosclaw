"""Bounded original STL facts, independent of ROS, filesystem and authority."""

import base64
import binascii
import hashlib
import json
import math
import re
import struct
from functools import lru_cache

MAX_BYTES = 8_000_000
MAX_TRIANGLES = 100_000


def stl_bounds(raw):
    """Read original vertices; no URI fetching, mesh repair, simplification or execution.

    Both STL encodings are supported. Binary solid-prefixed headers are detected
    by exact byte length, not a heuristic based on the first word. No trailing
    payload, nonfinite coordinate, truncated facet or unbounded count is accepted.
    """
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("bounded original STL bytes required")
    # Exact immutable byte keys; no URI/path/freshness or loaded-source state is
    # cached. Fresh dictionary results prevent callers mutating cached facts.
    return json.loads(_stl_bounds_json(raw))


@lru_cache(maxsize=8)
def _stl_bounds_json(raw):
    """At most 8 bounded byte keys (64MB); immutable offline geometry facts only."""
    bounds = [[math.inf, -math.inf] for _ in range(3)]
    count = 0

    def triangle(normal, vertices):
        nonlocal count
        values = tuple(normal) + tuple(v for vertex in vertices for v in vertex)
        if len(values) != 12 or not all(math.isfinite(v) for v in values):
            raise ValueError("finite complete original STL facet required")
        for vertex in vertices:
            for i, v in enumerate(vertex):
                bounds[i][0] = min(bounds[i][0], v)
                bounds[i][1] = max(bounds[i][1], v)
        count += 1
        if count > MAX_TRIANGLES:
            raise ValueError("original STL triangle budget exceeded")

    encoded_count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else None
    if encoded_count is not None and 84 + encoded_count * 50 == len(raw):
        if not 1 <= encoded_count <= MAX_TRIANGLES:
            raise ValueError("bounded nonempty binary STL triangle count required")
        for offset in range(84, len(raw), 50):
            values = struct.unpack_from("<12fH", raw, offset)
            triangle(values[:3], (values[3:6], values[6:9], values[9:12]))
        encoding = "BINARY_STL"
    else:
        try:
            text = raw.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("exact binary STL length or ASCII grammar required") from exc
        if "\x00" in text:
            raise ValueError("truncated or trailing binary STL is not ASCII")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or not (lines[0] == "solid" or lines[0].startswith("solid ")):
            raise ValueError("ASCII STL solid header required")
        if not (lines[-1] == "endsolid" or lines[-1].startswith("endsolid ")):
            raise ValueError("complete ASCII STL closing solid required")
        facets = lines[1:-1]
        if not facets or len(facets) % 7 or len(facets) // 7 > MAX_TRIANGLES:
            raise ValueError("bounded complete ASCII STL facet grammar required")

        def vector(line, prefix):
            parts = line.split()
            if parts[: len(prefix)] != prefix or len(parts) != len(prefix) + 3:
                raise ValueError("literal finite ASCII STL vector required")
            try:
                return tuple(float(x) for x in parts[len(prefix) :])
            except ValueError as exc:
                raise ValueError("literal numeric ASCII STL coordinates required") from exc

        for offset in range(0, len(facets), 7):
            facet = facets[offset : offset + 7]
            if facet[1] != "outer loop" or facet[5:] != ["endloop", "endfacet"]:
                raise ValueError("complete literal ASCII STL facet required")
            triangle(
                vector(facet[0], ["facet", "normal"]), [vector(v, ["vertex"]) for v in facet[2:5]]
            )
        encoding = "ASCII_STL"
    if count < 1 or not any(lo < hi for lo, hi in bounds):
        raise ValueError("nonempty finite spatial STL geometry required")
    return json.dumps(
        {
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "original_bytes": len(raw),
            "encoding": encoding,
            "triangle_count": count,
            "local_min": [b[0] for b in bounds],
            "local_max": [b[1] for b in bounds],
            "derivation": "extrema over every original vertex; no repaired or substituted geometry",
            "source_authentication": "NOT_ESTABLISHED_BY_HASH",
            "physical_loaded_geometry": "NOT_VERIFIED",
            "execution_authority": "NONE",
        },
        allow_nan=False,
    )


def review_collision_assets(originals):
    """Bind explicit URIs to exact finite original bytes, never resolve a path."""
    if type(originals) is not list or not 1 <= len(originals) <= 32:
        raise ValueError("1..32 explicit original collision assets required")
    result = {}
    total = 0
    for row in originals:
        if type(row) is not dict or set(row) != {"uri", "sha256", "content_base64"}:
            raise ValueError("exact URI SHA256 original-base64 asset fields required")
        uri, digest, content = row["uri"], row["sha256"], row["content_base64"]
        package_uri = (
            type(uri) is str
            and re.fullmatch(r"package://[a-z][a-z0-9_]*/[A-Za-z0-9_./-]+\.stl", uri)
            and not any(p in {"", ".", ".."} for p in uri.split("://", 1)[1].split("/"))
        )
        staged_uri = type(uri) is str and re.fullmatch(
            r"/assets/collision-resources/[0-9a-f]{64}\.stl", uri
        )
        if (
            type(uri) is not str
            or len(uri) > 2048
            or not (package_uri or staged_uri)
            or uri in result
        ):
            raise ValueError("unique concrete package or digest-named staged STL URI required")
        if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("original collision asset SHA256 required")
        if staged_uri and uri.rsplit("/", 1)[-1] != digest + ".stl":
            raise ValueError("digest-named staged STL URI differs from original SHA")
        if type(content) is not str or len(content) > ((MAX_BYTES + 2) // 3) * 4:
            raise ValueError("bounded original collision asset base64 required")
        try:
            raw = base64.b64decode(content, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("exact original collision asset base64 required") from exc
        if base64.b64encode(raw).decode() != content:
            raise ValueError("canonical original base64 required")
        total += len(raw)
        if total > 16_000_000 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("original asset hash or total byte budget differs")
        result[uri] = stl_bounds(raw)
    return result


def collision_assets_from_observations(observations):
    """Prepared resource bytes are explicitly not a ROS interface observation.

    The optional envelope retains original JSON bytes so an independent source
    reviewer can compare their SHA with the owned launch/capture asset inventory.
    Its presence establishes no source authenticity, readiness or authority.
    """
    import json

    envelope = observations.get("prepared_collision_assets")
    if envelope is None:
        return None
    if (
        type(envelope) is not dict
        or set(envelope) != {"schema_version", "source", "sha256", "document"}
        or envelope["schema_version"] != "rosclaw.prepared_collision_asset_observation.v1"
        or envelope["source"] != "/assets/collision-asset-originals.json"
        or type(envelope["document"]) is not str
    ):
        raise ValueError("closed explicit prepared source-resource observation required")
    raw = envelope["document"].encode("utf-8")
    if not 0 < len(raw) <= 5_000_000 or hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
        raise ValueError("original prepared collision manifest bytes differ")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("unique original collision JSON keys required")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("finite original collision JSON required")

    originals = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    facts = review_collision_assets(originals)
    return originals, facts, envelope["sha256"]


def staged_collision_resource_pins(observations):
    """Only exact digest-named staged STL files can enter source admission."""
    parsed = collision_assets_from_observations(observations)
    if parsed is None:
        return {}
    _originals, assets, document_sha = parsed
    pins = {"collision-asset-originals.json": document_sha}
    for uri, fact in assets.items():
        match = re.fullmatch(r"/assets/collision-resources/([0-9a-f]{64})\.stl", uri)
        if match is None or match[1] != fact["original_sha256"]:
            raise ValueError("source admission requires exact digest-named staged original STL URI")
        pins["collision-resources/" + match[1] + ".stl"] = fact["original_sha256"]
    return pins
