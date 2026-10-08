"""Anonymous synthetic URDFs; the actual held-out Body remains unseen."""

import hashlib
import math

import pytest

from rosclaw.connectors.ros.context.geometry import derive_collision_envelope


def robot(content):
    return f'<robot name="anonymous">{content}</robot>'.encode()


def box(name="base", size="0.4 0.2 0.1", origin=""):
    return f'<link name="{name}"><collision>{origin}<geometry><box size="{size}"/></geometry></collision></link>'


def joint(kind="fixed", origin="", limit="", parent="base", child="tool"):
    return f'<joint name="connection" type="{kind}"><parent link="{parent}"/><child link="{child}"/>{origin}{limit}</joint>'


def report(content, base="base"):
    return derive_collision_envelope(robot(content), base_frame=base)


def unknown(result):
    assert result["complete"] is False
    assert result["physical_radius_m"] is None
    assert result["unsupported_reasons"]
    assert result["capabilities_granted"] == []
    assert result["cleaning_attachment_inferred"] is False


def test_named_geometry_cannot_grant_binding_cleaner_or_drive():
    data = robot(box())
    result = derive_collision_envelope(data, base_frame="base")
    assert result["complete"] is True
    assert result["physical_radius_m"] == pytest.approx(math.hypot(0.2, 0.1))
    assert result["source_urdf_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["resolved_collision_count"] == result["collision_count"] == 1
    assert result["evidence_domain"] == "OFFLINE_URDF"
    assert result["evidence_role"] == "candidate_geometry_not_verified_binding"
    assert result["capabilities_granted"] == []
    assert not result["cleaning_attachment_inferred"]


def test_fixed_collision_rotation_and_translation():
    result = report(box(origin='<origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>'))
    assert result["physical_radius_m"] == pytest.approx(math.hypot(1.1, 0.2))


def test_base_frame_inverse_applies_to_root_and_child_collisions():
    result = report(
        box()
        + box("tool", "0.2 0.2 0.2")
        + joint(origin='<origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>'),
        base="tool",
    )
    assert result["physical_radius_m"] == pytest.approx(math.hypot(1.2, 0.1))


@pytest.mark.parametrize("kind", ["continuous", "revolute"])
def test_articulated_subtree_bounds_all_rotations(kind):
    result = report(
        box()
        + box("tool", "0.4 0.2 0.6", '<origin xyz="0.5 0 0"/>')
        + joint(kind, '<origin xyz="0.2 0.1 0"/>')
    )
    # Rotated 3D collision box is contained by this translated sphere.
    bound = math.hypot(0.2, 0.1) + math.hypot(0.7, 0.1, 0.3)
    assert result["physical_radius_m"] == pytest.approx(bound)
    # Independently sample all rotation directions around arbitrary axes.
    for yaw in range(0, 360, 7):
        a = math.radians(yaw)
        x = 0.2 + 0.7 * math.cos(a) - 0.1 * math.sin(a)
        y = 0.1 + 0.7 * math.sin(a) + 0.1 * math.cos(a)
        assert math.hypot(x, y) <= result["physical_radius_m"]


def test_prismatic_limits_and_nested_fixed_subtree_are_included():
    result = report(
        box()
        + '<link name="tool"/>'
        + box("tip", "0.2 0.2 0.2")
        + joint("prismatic", '<origin xyz="0.3 0 0"/>', '<limit lower="-0.4" upper="0.8"/>')
        + '<joint name="tip_mount" type="fixed"><parent link="tool"/><child link="tip"/><origin xyz="0.2 0 0"/></joint>'
    )
    assert result["physical_radius_m"] == pytest.approx(0.3 + 0.8 + 0.2 + math.sqrt(0.03))


@pytest.mark.parametrize(
    "shape", ['<sphere radius="0.2"/>', '<cylinder radius="0.2" length="0.5"/>']
)
def test_radial_primitives_use_conservative_containing_box(shape):
    result = report(f'<link name="base"><collision><geometry>{shape}</geometry></collision></link>')
    assert result["complete"]
    assert result["physical_radius_m"] >= 0.2


def test_any_unknown_mesh_discards_partial_numeric_guess():
    result = report(
        box()
        + '<link name="tool"><collision><geometry><mesh filename="package://unknown/asset.stl"/></geometry></collision></link>'
        + joint()
    )
    unknown(result)
    assert result["collision_count"] == 2
    assert result["resolved_collision_count"] == 1


@pytest.mark.parametrize(
    "content,base",
    [
        (box() + box(), "base"),
        (box() + box("tool"), "base"),
        (box(), "absent"),
        (
            '<link name="base"><visual><geometry><box size="1 1 1"/></geometry></visual></link>',
            "base",
        ),
        (box() + box("tool") + joint("continuous"), "tool"),
        (box() + box("tool") + joint("floating"), "base"),
        (box() + box("tool") + joint("prismatic"), "base"),
        (box() + box("tool") + joint("prismatic", limit='<limit lower="2" upper="1"/>'), "base"),
        (box(size="nan 1 1"), "base"),
        (box(size="-1 1 1"), "base"),
        (box(size="1e308 1e308 1e308", origin='<origin xyz="1.7e308 0 0" rpy="0 0 0.7"/>'), "base"),
        (box() + '<xacro:macro xmlns:xacro="urn:xacro" name="future"/>', "base"),
        (box() + '<link name="${unexpanded}"/>', "base"),
    ],
)
def test_invalid_or_incomplete_evidence_stays_unknown(content, base):
    unknown(report(content, base))


def test_entity_declarations_and_encoded_entities_are_rejected():
    data = '<!DOCTYPE robot [<!ENTITY x "0.4">]><robot name="r"><link name="base"><collision><geometry><box size="&x; 0.2 0.1"/></geometry></collision></link></robot>'
    for encoding in ("utf-8", "utf-16"):
        unknown(derive_collision_envelope(data.encode(encoding), base_frame="base"))


def test_explicit_bytes_and_bounded_size():
    with pytest.raises(TypeError):
        derive_collision_envelope("text", base_frame="base")
    unknown(derive_collision_envelope(b"x" * 5_000_001, base_frame="base"))


def test_disconnected_cycle_cannot_hide_collision():
    unknown(
        report(
            box()
            + box("tool")
            + box("tip")
            + joint(parent="tip")
            + '<joint name="cycle" type="fixed"><parent link="tool"/><child link="tip"/></joint>'
        )
    )
