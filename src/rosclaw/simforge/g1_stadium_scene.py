"""Deterministic native-MuJoCo football goal for qualified G1 scenes.

The qualified RoboNaldo scene ships with a heavy free box behind the target.
This module removes that box from a transient :class:`mujoco.MjSpec` and adds
a collision-capable training goal.  No external mesh or rendered pixel is
used for task scoring.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json

_SCENE_REL = Path("g1_description/scene_with_ball.xml")


@dataclass(frozen=True)
class G1TrainingGoalSpec:
    """Geometry and scoring contract for a humanoid-scale training goal."""

    plane_x_m: float = 5.0
    width_m: float = 2.4
    height_m: float = 1.6
    depth_m: float = 1.0
    post_radius_m: float = 0.035
    net_strand_radius_m: float = 0.003
    target_y_m: float = 1.0
    target_z_m: float = 0.115
    precision_radius_m: float = 0.16
    schema_version: str = "rosclaw.simforge.g1_training_goal_spec.v3"

    def __post_init__(self) -> None:
        values = (
            self.plane_x_m,
            self.width_m,
            self.height_m,
            self.depth_m,
            self.post_radius_m,
            self.net_strand_radius_m,
            self.target_y_m,
            self.target_z_m,
            self.precision_radius_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("training goal values must be finite")
        if not 4.0 <= self.plane_x_m <= 12.0:
            raise ValueError("training goal plane must be in [4, 12] m")
        if not 1.5 <= self.width_m <= 7.32:
            raise ValueError("training goal width must be in [1.5, 7.32] m")
        if not 1.0 <= self.height_m <= 2.44:
            raise ValueError("training goal height must be in [1.0, 2.44] m")
        if not 0.4 <= self.depth_m <= 2.0:
            raise ValueError("training goal depth must be in [0.4, 2.0] m")
        if not 0.02 <= self.post_radius_m <= 0.08:
            raise ValueError("training goal post radius must be in [0.02, 0.08] m")
        if abs(self.target_y_m) >= self.width_m / 2.0 - self.post_radius_m:
            raise ValueError("training target must remain inside the goal posts")
        if not 0.115 <= self.target_z_m < self.height_m - self.post_radius_m:
            raise ValueError("training target height must remain inside the goal")
        if not 0.05 <= self.precision_radius_m <= 0.30:
            raise ValueError("precision radius must be in [0.05, 0.30] m")

    @property
    def spec_hash(self) -> str:
        return hash_json(asdict(self))

    @property
    def target_corner(self) -> str:
        """Declared corner using the kicker-facing +x, +y-is-left convention."""

        side = "left" if self.target_y_m >= 0.0 else "right"
        level = "upper" if self.target_z_m >= self.height_m / 2.0 else "lower"
        return f"{side}_{level}"

    @property
    def target_corner_center_m(self) -> tuple[float, float, float]:
        ball_radius = 0.115
        y = math.copysign(self.width_m / 2.0 - ball_radius, self.target_y_m)
        z = self.height_m - ball_radius if "upper" in self.target_corner else ball_radius
        return (self.plane_x_m, y, z)


def build_g1_stadium_model(asset_root: Path, spec: G1TrainingGoalSpec | None = None) -> Any:
    """Compile a qualified G1 scene with the wall replaced by a native goal."""

    import mujoco

    goal = spec or G1TrainingGoalSpec()
    scene = asset_root.expanduser().resolve() / _SCENE_REL
    parent = mujoco.MjSpec.from_file(str(scene))
    wall = parent.body("box")
    if wall is None:
        raise ValueError("qualified G1 scene does not contain the replaceable box body")
    parent.delete(wall)
    _style_pitch_and_ball(parent)
    _add_goal(parent, goal)
    model = parent.compile()
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box") >= 0:
        raise AssertionError("stadium scene retained the original wall")
    for name in ("goal_left_post", "goal_right_post", "goal_crossbar", "goal_back_net"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            raise AssertionError(f"stadium scene is missing {name}")
    return model


def g1_stadium_scene_hash(asset_root: Path, spec: G1TrainingGoalSpec | None = None) -> str:
    """Bind the derived scene to both its source XML and declarative goal spec."""

    goal = spec or G1TrainingGoalSpec()
    scene = asset_root.expanduser().resolve() / _SCENE_REL
    return hash_json(
        {
            "source_scene_hash": hash_bytes(scene.read_bytes()),
            "goal_spec_hash": goal.spec_hash,
            "builder_hash": hash_bytes(Path(__file__).read_bytes()),
        }
    )


def _add_goal(parent: Any, spec: G1TrainingGoalSpec) -> None:
    import mujoco

    world = parent.worldbody
    x = spec.plane_x_m
    half_width = spec.width_m / 2.0
    radius = spec.post_radius_m
    white = (0.98, 0.98, 0.96, 1.0)

    def capsule(
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        *,
        rgba: tuple[float, float, float, float] = white,
        collision: bool = True,
        strand: bool = False,
        custom_radius: float | None = None,
    ) -> None:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(*start, *end),
            size=(
                custom_radius
                if custom_radius is not None
                else spec.net_strand_radius_m
                if strand
                else radius,
                0.0,
                0.0,
            ),
            rgba=rgba,
            contype=1 if collision else 0,
            conaffinity=1 if collision else 0,
        )

    capsule("goal_left_post", (x, -half_width, 0.0), (x, -half_width, spec.height_m))
    capsule("goal_right_post", (x, half_width, 0.0), (x, half_width, spec.height_m))
    capsule("goal_crossbar", (x, -half_width, spec.height_m), (x, half_width, spec.height_m))
    rear_bottom_x = x + spec.depth_m
    rear_top_x = x + spec.depth_m * 0.68
    support_radius = radius * 0.58
    capsule(
        "goal_left_back_post",
        (rear_bottom_x, -half_width, 0.0),
        (rear_top_x, -half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_right_back_post",
        (rear_bottom_x, half_width, 0.0),
        (rear_top_x, half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_left_depth_bar",
        (x, -half_width, spec.height_m),
        (rear_top_x, -half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_right_depth_bar",
        (x, half_width, spec.height_m),
        (rear_top_x, half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_back_crossbar",
        (rear_top_x, -half_width, spec.height_m),
        (rear_top_x, half_width, spec.height_m),
        custom_radius=support_radius,
    )
    capsule(
        "goal_back_ground_bar",
        (rear_bottom_x, -half_width, 0.025),
        (rear_bottom_x, half_width, 0.025),
        custom_radius=support_radius,
    )

    # A fine sloped net is visual geometry. A deterministic compliant force
    # field in the rollout models capture without the rigid-wall rebound of a
    # transparent box.
    net = (0.91, 0.94, 0.92, 0.74)
    vertical_count = 17
    horizontal_count = 11
    for index in range(vertical_count):
        y = -half_width + spec.width_m * index / (vertical_count - 1)
        capsule(
            "goal_back_net" if index == vertical_count // 2 else f"goal_back_net_v_{index}",
            (rear_bottom_x, y, 0.03),
            (rear_top_x, y, spec.height_m),
            rgba=net,
            collision=False,
            strand=True,
        )
    for index in range(horizontal_count):
        z = spec.height_m * index / (horizontal_count - 1)
        rear_x = rear_bottom_x + (rear_top_x - rear_bottom_x) * z / spec.height_m
        capsule(
            f"goal_back_net_h_{index}",
            (rear_x, -half_width, max(0.025, z)),
            (rear_x, half_width, max(0.025, z)),
            rgba=net,
            collision=False,
            strand=True,
        )
    for side, y in (("left", -half_width), ("right", half_width)):
        for index in range(7):
            z = spec.height_m * index / 6.0
            rear_x = rear_bottom_x + (rear_top_x - rear_bottom_x) * z / spec.height_m
            capsule(
                f"goal_{side}_net_h_{index}",
                (x, y, max(0.025, z)),
                (rear_x, y, max(0.025, z)),
                rgba=net,
                collision=False,
                strand=True,
            )
        for index in range(6):
            alpha = index / 5.0
            net_x_bottom = x + alpha * (rear_bottom_x - x)
            net_x_top = x + alpha * (rear_top_x - x)
            capsule(
                f"goal_{side}_net_v_{index}",
                (net_x_bottom, y, 0.025),
                (net_x_top, y, spec.height_m),
                rgba=net,
                collision=False,
                strand=True,
            )
    for index in range(9):
        y = -half_width + spec.width_m * index / 8.0
        capsule(
            f"goal_roof_net_{index}",
            (x, y, spec.height_m),
            (rear_top_x, y, spec.height_m),
            rgba=net,
            collision=False,
            strand=True,
        )


def _style_pitch_and_ball(parent: Any) -> None:
    """Replace the blue grid with a pitch and add a lightweight ball pattern."""

    import mujoco

    floor = parent.geom("floor")
    floor.material = ""
    floor.rgba = (0.055, 0.24, 0.075, 1.0)
    world = parent.worldbody
    line = (0.93, 0.94, 0.90, 0.92)
    for name, pos, size in (
        ("pitch_goal_line", (5.0, 0.0, 0.004), (0.018, 4.2, 0.003)),
        ("pitch_box_front", (2.8, 0.0, 0.004), (0.018, 2.8, 0.003)),
        ("pitch_box_left", (3.9, -2.8, 0.004), (1.1, 0.018, 0.003)),
        ("pitch_box_right", (3.9, 2.8, 0.004), (1.1, 0.018, 0.003)),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=line,
            contype=0,
            conaffinity=0,
        )
    ball = parent.body("ball")
    for index, position in enumerate(
        (
            (0.102, 0.0, 0.0),
            (-0.102, 0.0, 0.0),
            (0.0, 0.102, 0.0),
            (0.0, -0.102, 0.0),
            (0.0, 0.0, 0.102),
        )
    ):
        ball.add_geom(
            name=f"ball_patch_{index}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=position,
            size=(0.026, 0.0, 0.0),
            rgba=(0.025, 0.025, 0.025, 1.0),
            density=0.0,
            contype=0,
            conaffinity=0,
        )


__all__ = [
    "G1TrainingGoalSpec",
    "build_g1_stadium_model",
    "g1_stadium_scene_hash",
]
