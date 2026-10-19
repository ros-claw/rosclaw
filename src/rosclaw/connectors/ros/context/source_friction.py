"""Pure source facts and adaptation for offline Core proposals; no SDK execution."""

import hashlib
import math
import re
import xml.etree.ElementTree as ET

_FRAME_URIS = {"http://ignitionrobotics.org/schema", "http://gazebosim.org/schemas"}


def parse_source_xml(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 4_000_000:
        raise ValueError("bounded original UTF-8 XML bytes required")
    return ET.fromstring(raw.decode("utf-8"))


def source_friction_fact(link, ode, names):
    direction = ode.find("fdir1")
    if direction is None:
        return None
    attrs = list(direction.attrib.items())
    if len(attrs) != 1:
        raise ValueError("one explicit namespaced original friction frame required")
    key, frame = attrs[0]
    if not key.startswith("{") or not key.endswith("}expressed_in"):
        raise ValueError("unsupported friction frame attribute")
    uri = key[1:].split("}", 1)[0]
    if uri not in _FRAME_URIS or frame not in names:
        raise ValueError("unresolved original friction direction frame")
    text = (direction.text or "").split()
    if len(text) != 3:
        raise ValueError("three original friction direction components required")
    values = [float(x) for x in text]
    coefficients = [float(ode.findtext(name, "nan")) for name in ["mu", "mu2"]]
    if (
        not all(math.isfinite(x) for x in values + coefficients)
        or not any(x != 0 for x in values)
        or any(x < 0 for x in coefficients)
    ):
        raise ValueError("finite original direction and nonnegative friction required")
    return (link, uri, frame, tuple(values), tuple(coefficients))


SDK_IMAGE = "sha256:8cc3fbf1eaa57058f0c1386c061253a55651106bd46b98fb046439f966c9019b"


def prepare_source_friction_frame(
    urdf_bytes, world_sdf_bytes, *, source_urdf_sha256, robot_entity, sdk_image
):
    """Return stage-only XML and a derivation; never grant physics acceptance."""
    if sdk_image != SDK_IMAGE:
        raise ValueError("only the independently inspected fixed SDK parser is supported")
    urdf, world = parse_source_xml(urdf_bytes), parse_source_xml(world_sdf_bytes)
    if hashlib.sha256(urdf_bytes).hexdigest() != source_urdf_sha256:
        raise ValueError("original URDF digest differs")
    if urdf.tag != "robot" or world.tag != "sdf" or not isinstance(robot_entity, str):
        raise ValueError("original URDF and one declared SDF entity required")
    names = {x.get("name") for x in urdf.findall("link")}
    source = []
    for block in urdf.findall("gazebo"):
        for ode in block.findall("collision/surface/friction/ode"):
            fact = source_friction_fact(block.get("reference"), ode, names)
            if fact is not None:
                if fact[0] not in names:
                    raise ValueError("friction source references an unknown link")
                source.append(fact)
    models = [x for x in world.findall("world/model") if x.get("name") == robot_entity]
    if len(models) != 1:
        raise ValueError("one declared owned source model required")
    world_links = {x.get("name") for x in models[0].findall("link")}
    if any(fact[2] not in world_links for fact in source):
        raise ValueError("friction frame has no loaded SDF link")
    actual = []
    for link in models[0].findall("link"):
        for ode in link.findall("collision/surface/friction/ode"):
            fact = source_friction_fact(link.get("name"), ode, names)
            if fact is not None:
                actual.append(fact)
    if not source or len(source) > 128 or sorted(source) != sorted(actual):
        raise ValueError("derived friction directions/frames/values differ from original URDF")
    uris = {x[1] for x in source}
    if len(uris) != 1:
        raise ValueError("a single original friction namespace is required")
    uri = next(iter(uris))
    text = world_sdf_bytes.decode("utf-8")
    tags = re.findall(r"<fdir1\b[^<>]*>", text)
    if len(tags) != len(actual):
        raise ValueError("unexpected additional or differently qualified friction declarations")
    spelling = re.compile(r"(?<![\w:.-])([A-Za-z_][\w.-]*):expressed_in(?=\s*=)")
    if any(len(spelling.findall(tag)) != 1 for tag in tags):
        raise ValueError("one literal prefixed frame attribute per direction required")
    changed = sum(spelling.search(tag).group(1) != "gz" for tag in tags)
    after = text
    if changed:
        root = re.search(r"<sdf\b[^<>]*>", after)
        if root is None:
            raise ValueError("one SDF root start tag required")
        binding = re.search(r"\bxmlns:gz\s*=\s*(['\"])(.*?)\1", root.group())
        if binding and binding.group(2) != uri:
            raise ValueError("gz prefix is already bound to a different namespace")
        if binding is None:
            start = root.group()[:-1] + f' xmlns:gz="{uri}">'
            after = after[: root.start()] + start + after[root.end() :]
        after = re.sub(
            r"<fdir1\b[^<>]*>", lambda match: spelling.sub("gz:expressed_in", match.group()), after
        )
    if ET.canonicalize(text, rewrite_prefixes=True) != ET.canonicalize(
        after, rewrite_prefixes=True
    ):
        raise ValueError("namespace adaptation changed expanded XML semantics")
    if any(spelling.search(tag).group(1) != "gz" for tag in re.findall(r"<fdir1\b[^<>]*>", after)):
        raise ValueError("literal backend friction attribute was not preserved")
    raw = after.encode("utf-8")
    return raw, {
        "schema_version": "rosclaw.sdk_source_friction_frame_compatibility.v1",
        "status": "CONDITIONAL_XML_SPELLING_ADAPTED_NOT_PHYSICS_VERIFIED"
        if changed
        else "ALREADY_LITERAL_COMPATIBLE_NOT_PHYSICS_VERIFIED",
        "SDK_image": sdk_image,
        "source_urdf_sha256": source_urdf_sha256,
        "robot_entity": robot_entity,
        "original_world_sha256": hashlib.sha256(world_sdf_bytes).hexdigest(),
        "adapted_world_sha256": hashlib.sha256(raw).hexdigest(),
        "friction_frames_checked": len(actual),
        "attribute_spellings_changed": changed,
        "original_friction_facts": source,
        "expanded_XML_semantics_equal": True,
        "geometry_friction_numbers_URDF_and_Body_limits_changed": False,
        "runtime_friction_interpretation": "REQUIRES_NEW_OWNED_WORLD_PROOF",
        "current_authority": "NONE",
        "scope": "EXPLICIT_OPERATOR_OWNED_DISPOSABLE_SIM_ONLY",
    }
