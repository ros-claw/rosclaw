"""Synthetic observer contracts; no physical or Native acceptance claim."""

import pytest

from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting
from rosclaw.connectors.ros.verification.occupancy_geometry import (
    ModelPose,
    OccupancyProjector,
    seal_obstacle_geometry,
)


def sdf(shape="<sphere><radius>0.1</radius></sphere>", extra=""):
    return (
        '<sdf version="1.9"><world name="room"><model name="pedestrian">'
        + extra
        + '<link name="body"><collision name="contact"><geometry>'
        + shape
        + "</geometry></collision></link></model></world></sdf>"
    ).encode()


def setup():
    grid = CoverageVerifier(
        width=4,
        height=1,
        resolution=1,
        accessible_cells=[0, 1, 2, 3],
        cleaning_polygon=[(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)],
    )
    geometry = seal_obstacle_geometry(sdf(), obstacle_names=("pedestrian",))
    return grid, OccupancyProjector(grid, geometry)


def packet(projector, x, time=1, sequence=0, **changes):
    kwargs = {
        "run_id": "run",
        "mission_id": "mission",
        "sequence": sequence,
        "frame_id": "map",
        "sim_time_sec": time,
        "ground_truth_age_sec": 0.01,
        "complete": True,
    }
    kwargs.update(changes)
    return projector.project((ModelPose("pedestrian", x, 0.5, time),), **kwargs)


def test_sealed_collision_geometry_and_same_packet_occupancy_block_until_actual_revisit():
    grid, projector = setup()
    accounting = OccupancyAccounting(
        grid, run_id="run", mission_id="mission", geometry_hash=projector.geometry_hash
    )
    s = packet(projector, 0.5)
    assert s.occupied_cells == (0,)
    accounting.observe(CleaningPose(0.5, 0.5, 0, 1, True), s, artifact_hash=s.artifact_hash())
    assert grid.visits == {}
    s = packet(projector, 20, time=2, sequence=1)
    accounting.observe(CleaningPose(1.5, 0.5, 0, 2, True), s, artifact_hash=s.artifact_hash())
    assert set(grid.visits) == {1}
    s = packet(projector, 20, time=3, sequence=2)
    accounting.observe(CleaningPose(0.5, 0.5, 0, 3, True), s, artifact_hash=s.artifact_hash())
    assert set(grid.visits) == {0, 1}
    assert accounting.result()["fixed_denominator_cells"] == 4


def test_cell_edge_contact_is_conservatively_blocked_even_if_center_outside_shape():
    _, projector = setup()
    assert packet(projector, 1).occupied_cells == (0, 1)


@pytest.mark.parametrize(
    "shape,minimum",
    [
        ("<box><size>2 4 6</size></box>", 3.74),
        ("<cylinder><radius>2</radius><length>4</length></cylinder>", 2.82),
        ("<sphere><radius>2</radius></sphere>", 1.99),
    ],
)
def test_all_collision_primitives_are_enclosed(shape, minimum):
    geometry = seal_obstacle_geometry(sdf(shape), obstacle_names=("pedestrian",))
    assert geometry.model_radii[0][1] > minimum
    assert len(geometry.sdf_sha256) == 64


def test_link_and_collision_offsets_are_added_without_rotation_underestimate():
    data = sdf().replace(b'<link name="body">', b'<link name="body"><pose>3 4 0 0 0 2</pose>')
    data = data.replace(
        b'<collision name="contact">', b'<collision name="contact"><pose>0 0 2 0 1 0</pose>'
    )
    geometry = seal_obstacle_geometry(data, obstacle_names=("pedestrian",))
    assert geometry.model_radii == (("pedestrian", 7.1),)


@pytest.mark.parametrize(
    "data",
    [
        sdf("<mesh><uri>model://person/mesh.dae</uri></mesh>"),
        sdf("<sphere><radius>nan</radius></sphere>"),
        sdf("<sphere><radius>1</radius><radius>8</radius></sphere>"),
        sdf("<box><size>1e300 1 1</size></box>"),
        sdf().replace(
            b"<geometry>", b"<geometry><sphere><radius>8</radius></sphere></geometry><geometry>"
        ),
        sdf().replace(
            b'<link name="body">',
            b'<link name="body"><pose>0 0 0 0 0 0</pose><pose>9 0 0 0 0 0</pose>',
        ),
        sdf("<box><size>1 -2 3</size></box>"),
        sdf("<cylinder><radius>1</radius><length>0</length></cylinder>"),
        sdf(extra='<joint name="articulation" type="revolute"/>'),
        sdf(extra="<include><uri>model://unknown</uri></include>"),
        sdf().replace(
            b'<link name="body">',
            b'<link name="body"><pose relative_to="unknown">0 0 0 0 0 0</pose>',
        ),
        sdf().replace(b"<geometry>", b"<geometry/><visual>"),
        sdf()
        .replace(b"</link>", b'</link><link name="empty"/>')
        .replace(b'<collision name="contact">', b'<visual name="contact">')
        .replace(b"</collision>", b"</visual>"),
        b'<!DOCTYPE sdf [<!ENTITY e "1">]>' + sdf(),
        sdf().decode().encode("utf-16"),
    ],
)
def test_unresolved_or_invalid_collision_geometry_never_means_empty_occupancy(data):
    with pytest.raises(ValueError):
        seal_obstacle_geometry(data, obstacle_names=("pedestrian",))


@pytest.mark.parametrize(
    "changes",
    [
        {"complete": False},
        {"frame_id": "odom"},
        {"ground_truth_age_sec": 0.3},
        {"sim_time_sec": 2},
        {"sequence": True},
        {"run_id": ""},
    ],
)
def test_incomplete_stale_or_misaligned_packet_returns_no_snapshot(changes):
    _, projector = setup()
    with pytest.raises(ValueError):
        packet(projector, 0.5, **changes)


@pytest.mark.parametrize(
    "poses",
    [
        (),
        (ModelPose("other", 0, 0, 1),),
        (ModelPose("pedestrian", float("nan"), 0, 1),),
        (ModelPose("pedestrian", 0, 0, 2),),
        (ModelPose("pedestrian", 0, 0, 1), ModelPose("pedestrian", 0, 0, 1)),
    ],
)
def test_missing_ambiguous_nonfinite_or_different_time_models_fail_closed(poses):
    _, projector = setup()
    with pytest.raises(ValueError):
        projector.project(
            poses,
            run_id="run",
            mission_id="mission",
            sequence=0,
            frame_id="map",
            sim_time_sec=1,
            ground_truth_age_sec=0.01,
            complete=True,
        )


def test_geometry_hash_binds_exact_source_bytes_and_projection_rejects_changed_denominator():
    grid, projector = setup()
    changed = seal_obstacle_geometry(sdf().replace(b"0.1", b"0.2"), obstacle_names=("pedestrian",))
    assert changed.artifact_hash() != projector.geometry_hash
    grid.accessible.remove(0)
    with pytest.raises(ValueError, match="denominator"):
        packet(projector, 20)


def test_projection_work_budget_returns_no_partial_free_or_occupied_snapshot():
    grid = CoverageVerifier(
        width=500,
        height=500,
        resolution=0.001,
        accessible_cells=[0],
        cleaning_polygon=[(-0.1, -0.1), (0.1, -0.1), (0, 0.1)],
    )
    geometry = seal_obstacle_geometry(
        sdf("<sphere><radius>2</radius></sphere>"), obstacle_names=("pedestrian",)
    )
    projector = OccupancyProjector(grid, geometry)
    with pytest.raises(ValueError, match="budget"):
        packet(projector, 0.25)
