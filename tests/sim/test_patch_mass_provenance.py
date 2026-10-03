"""Model patch roundtrip keeps authored mass evidence and compiled semantics."""

import xml.etree.ElementTree as ET

import mujoco
import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend


def load(tmp_path, declaration):
    xml = (
        '<mujoco><worldbody><body name="box"><freejoint/>'
        f'<geom name="g" type="box" size=".06 .06 .06" {declaration}/>'
        "</body></worldbody></mujoco>"
    )
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(xml, source={"kind": "fixture"})
    return backend, ref.model_ref


def patch(backend, ref, field, value):
    return backend.patch_model(
        ref,
        [{"op": "set", "target": {"type": "geom", "name": "g"}, "field": field, "value": value}],
    ).new_model_ref


@pytest.mark.parametrize("declaration", ['mass="1.728"', 'density="1000"'])
def test_unrelated_patch_preserves_default_equivalent_explicit_source(tmp_path, declaration):
    backend, ref = load(tmp_path, declaration)
    parent = backend.store.resolve(ref).read_bytes()
    child = patch(backend, ref, "rgba", [1, 0, 0, 1])
    assert backend.audit(child, checks=["A02_explicit_mass"]).status == "PASS"
    assert backend.store.resolve(ref).read_bytes() == parent
    manifest = backend.store.get(child)
    model = mujoco.MjModel.from_xml_string(manifest["mjcf_xml"])
    assert model.body_mass[1] == pytest.approx(1.728)


@pytest.mark.parametrize("field,value,expected", [("mass", 1.728, 1.728), ("density", 1000, 1.728)])
def test_patch_repairs_implicit_source_without_losing_declaration(tmp_path, field, value, expected):
    backend, ref = load(tmp_path, "")
    assert backend.audit(ref, checks=["A02_explicit_mass"]).status == "FAIL"
    child = patch(backend, ref, field, value)
    assert backend.audit(child, checks=["A02_explicit_mass"]).status == "PASS"
    geom = ET.fromstring(backend.store.get(child)["mjcf_xml"]).find(".//geom")
    assert geom.get(field) is not None
    assert mujoco.MjModel.from_xml_string(backend.store.get(child)["mjcf_xml"]).body_mass[
        1
    ] == pytest.approx(expected)


def test_unrelated_patch_does_not_fabricate_explicit_mass_for_implicit_model(tmp_path):
    backend, ref = load(tmp_path, "")
    child = patch(backend, ref, "rgba", [1, 0, 0, 1])
    assert backend.audit(child, checks=["A02_explicit_mass"]).status == "FAIL"


def test_density_patch_overrides_prior_mass_and_mass_can_override_density_again(tmp_path):
    backend, ref = load(tmp_path, 'mass="2"')
    child = patch(backend, ref, "density", 500)
    model = mujoco.MjModel.from_xml_string(backend.store.get(child)["mjcf_xml"])
    assert model.body_mass[1] == pytest.approx(0.864)
    grandchild = patch(backend, child, "mass", 3)
    model = mujoco.MjModel.from_xml_string(backend.store.get(grandchild)["mjcf_xml"])
    assert model.body_mass[1] == pytest.approx(3)
    assert backend.audit(grandchild, checks=["A02_explicit_mass"]).status == "PASS"
