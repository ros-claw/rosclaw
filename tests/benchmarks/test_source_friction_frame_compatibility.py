"""Pure namespace ABI checks; never create an SDK container or physics World."""

import hashlib
import xml.etree.ElementTree as ET

import pytest

from integrations.ros_probe.acceptance.simulation_source_friction_frame import (
    SDK_IMAGE,
    prepare_source_friction_frame,
)

URI = "http://ignitionrobotics.org/schema"


def fixture(prefix="ns0", robot="unrelated_mobile_platform", frame="base"):
    declaration = f'xmlns:{prefix}="{URI}"'
    ode = f'<ode><mu>1</mu><mu2>0</mu2><fdir1 {prefix}:expressed_in="{frame}">1 -1 0</fdir1></ode>'
    friction = f"<surface><friction>{ode}</friction></surface>"
    urdf = (
        f'<robot name="source" {declaration}><link name="base"/><link name="wheel"/>'
        f'<gazebo reference="wheel"><collision>{friction}</collision></gazebo></robot>'
    ).encode()
    world = (
        f'<sdf version="1.9" {declaration}><world name="owned">'
        f'<model name="{robot}"><link name="base"/><link name="wheel">'
        f'<collision name="wheel"><geometry><sphere><radius>0.1</radius></sphere></geometry>'
        f"{friction}</collision></link></model></world></sdf>"
    ).encode()
    return urdf, world


def adapt(urdf, world, **options):
    args = {
        "source_urdf_sha256": hashlib.sha256(urdf).hexdigest(),
        "robot_entity": "unrelated_mobile_platform",
        "sdk_image": SDK_IMAGE,
    }
    args.update(options)
    return prepare_source_friction_frame(urdf, world, **args)


@pytest.mark.parametrize("prefix", ["ns0", "ignition", "arbitrary_source_alias"])
def test_only_literal_spelling_changes_with_same_expanded_source_meaning(prefix):
    urdf, world = fixture(prefix)
    after, review = adapt(urdf, world)
    assert f'{prefix}:expressed_in="base"'.encode() in world
    assert b'gz:expressed_in="base"' in after
    assert b"<mu2>0</mu2>" in after
    assert ET.canonicalize(world.decode(), rewrite_prefixes=True) == ET.canonicalize(
        after.decode(), rewrite_prefixes=True
    )
    assert review["attribute_spellings_changed"] == 1
    assert review["source_urdf_sha256"] == hashlib.sha256(urdf).hexdigest()
    assert review["current_authority"] == "NONE"
    assert review["runtime_friction_interpretation"] == "REQUIRES_NEW_OWNED_WORLD_PROOF"
    assert review["geometry_friction_numbers_URDF_and_Body_limits_changed"] is False


def test_already_compatible_bytes_and_second_application_are_unchanged():
    urdf, world = fixture("gz")
    after, review = adapt(urdf, world)
    assert after == world and review["attribute_spellings_changed"] == 0
    assert adapt(urdf, after)[0] == after


def test_independent_entity_name_is_bound_without_robot_branch():
    urdf, world = fixture(robot="second_body")
    assert adapt(urdf, world, robot_entity="second_body")[1]["robot_entity"] == "second_body"
    with pytest.raises(ValueError, match="declared owned source model"):
        adapt(urdf, world)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"sdk_image": "sha256:" + "0" * 64}, "fixed SDK"),
        ({"source_urdf_sha256": "0" * 64}, "URDF digest"),
    ],
)
def test_unpinned_sdk_or_original_cannot_be_adapted(options, reason):
    with pytest.raises(ValueError, match=reason):
        adapt(*fixture(), **options)


@pytest.mark.parametrize("old,new", [(b"1 -1 0", b"1 1 0"), (b"<mu2>0", b"<mu2>0.1")])
def test_changed_physics_numbers_are_rejected(old, new):
    urdf, world = fixture()
    with pytest.raises(ValueError, match="differ from original URDF"):
        adapt(urdf, world.replace(old, new))


def test_missing_source_link_frame_cannot_become_loadable():
    with pytest.raises(ValueError, match="unresolved original"):
        adapt(*fixture(frame="invented"))
    urdf, world = fixture()
    with pytest.raises(ValueError, match="loaded SDF link"):
        adapt(urdf, world.replace(b'<link name="base"/>', b""))


def test_unqualified_frame_is_rejected_instead_of_guessing_its_meaning():
    urdf, world = fixture()
    with pytest.raises(ValueError, match="unsupported friction frame"):
        adapt(urdf.replace(b"ns0:expressed_in", b"expressed_in"), world)


def test_conflicting_gz_namespace_is_not_overwritten():
    urdf, world = fixture()
    world = world.replace(b'<sdf version="1.9"', b'<sdf version="1.9" xmlns:gz="urn:other"')
    with pytest.raises(ValueError, match="different namespace"):
        adapt(urdf, world)


def test_extra_unbound_friction_declarations_are_rejected():
    urdf, world = fixture()
    extra = b'<model name="other"><link name="base"><fdir1 ns0:expressed_in="base">1 -1 0</fdir1></link></model>'
    with pytest.raises(ValueError, match="unexpected additional"):
        adapt(urdf, world.replace(b"</world>", extra + b"</world>"))


def test_no_source_declaration_is_not_a_verification_success():
    urdf, world = fixture()
    with pytest.raises(ValueError, match="differ from original URDF"):
        adapt(urdf[: urdf.index(b"<gazebo")] + b"</robot>", world)
