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
_MODEL_REL = Path("g1_description/g1_liao.xml")


@dataclass(frozen=True)
class G1TrainingGoalSpec:
    """Geometry and scoring contract for a humanoid-scale training goal."""

    plane_x_m: float = 5.0
    width_m: float = 2.4
    height_m: float = 1.6
    depth_m: float = 1.0
    post_radius_m: float = 0.035
    net_strand_radius_m: float = 0.00125
    target_y_m: float = 1.0
    target_z_m: float = 0.115
    precision_radius_m: float = 0.16
    ball_free_joint_damping_n_s_m: float = 0.02
    ball_radius_m: float = 0.115
    ball_mass_kg: float = 0.41
    ball_contact_sliding_friction: float = 0.05
    ball_sliding_friction: float = 0.05
    ball_torsional_friction: float = 0.0003
    ball_rolling_friction: float = 0.00002
    regulation_field_enabled: bool = False
    field_length_m: float = 105.0
    field_width_m: float = 68.0
    field_line_width_m: float = 0.10
    goal_area_depth_m: float = 5.50
    penalty_area_depth_m: float = 16.50
    penalty_mark_distance_m: float = 11.0
    schema_version: str = "rosclaw.simforge.g1_training_goal_spec.v7"

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
            self.ball_free_joint_damping_n_s_m,
            self.ball_radius_m,
            self.ball_mass_kg,
            self.ball_contact_sliding_friction,
            self.ball_sliding_friction,
            self.ball_torsional_friction,
            self.ball_rolling_friction,
            self.field_length_m,
            self.field_width_m,
            self.field_line_width_m,
            self.goal_area_depth_m,
            self.penalty_area_depth_m,
            self.penalty_mark_distance_m,
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
        if not 0.001 <= self.net_strand_radius_m <= 0.006:
            raise ValueError("training goal net strand radius must be in [0.001, 0.006] m")
        if abs(self.target_y_m) > self.width_m / 2.0 - self.ball_radius_m:
            raise ValueError("training target must keep the whole ball inside the goal posts")
        if not self.ball_radius_m <= self.target_z_m <= self.height_m - self.ball_radius_m:
            raise ValueError("training target must keep the whole ball inside the goal height")
        if not 0.05 <= self.precision_radius_m <= 0.30:
            raise ValueError("precision radius must be in [0.05, 0.30] m")
        if not 0.001 <= self.ball_free_joint_damping_n_s_m <= 0.10:
            raise ValueError("ball free-joint damping must be in [0.001, 0.10] N s/m")
        if not 0.105 <= self.ball_radius_m <= 0.115:
            raise ValueError("football radius must be in [0.105, 0.115] m")
        if not 0.40 <= self.ball_mass_kg <= 0.46:
            raise ValueError("football mass must be in [0.40, 0.46] kg")
        if not 0.03 <= self.ball_contact_sliding_friction <= 0.80:
            raise ValueError("football contact sliding friction must be in [0.03, 0.80]")
        if not 0.03 <= self.ball_sliding_friction <= 0.80:
            raise ValueError("football-ground sliding friction must be in [0.03, 0.80]")
        if not 0.0 <= self.ball_torsional_friction <= 0.02:
            raise ValueError("football torsional friction must be in [0, 0.02]")
        if not 0.0 <= self.ball_rolling_friction <= 0.001:
            raise ValueError("football rolling friction must be in [0, 0.001]")
        if not isinstance(self.regulation_field_enabled, bool):
            raise ValueError("regulation field flag must be boolean")
        if self.regulation_field_enabled:
            if not 100.0 <= self.field_length_m <= 110.0:
                raise ValueError("international field length must be in [100, 110] m")
            if not 64.0 <= self.field_width_m <= 75.0:
                raise ValueError("international field width must be in [64, 75] m")
            if not 0.0 < self.field_line_width_m <= 0.12:
                raise ValueError("field line width must be in (0, 0.12] m")
            if (
                self.goal_area_depth_m != 5.50
                or self.penalty_area_depth_m != 16.50
                or self.penalty_mark_distance_m != 11.0
            ):
                raise ValueError("regulation penalty geometry does not match IFAB dimensions")

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
        y = math.copysign(self.width_m / 2.0 - self.ball_radius_m, self.target_y_m)
        z = (
            self.height_m - self.ball_radius_m
            if "upper" in self.target_corner
            else self.ball_radius_m
        )
        return (self.plane_x_m, y, z)


def build_g1_stadium_model(asset_root: Path, spec: G1TrainingGoalSpec | None = None) -> Any:
    """Compile a qualified G1 scene with the wall replaced by a native goal."""

    goal = spec or G1TrainingGoalSpec()
    parent = _stadium_spec(asset_root, goal)
    model = parent.compile()
    _require_stadium_model(model)
    return model


def build_g1_coupled_stadium_model(
    asset_root: Path,
    *,
    passer_origin_m: tuple[float, float, float],
    spec: G1TrainingGoalSpec | None = None,
) -> Any:
    """Compile the two-G1 replay scene with the same native football goal.

    This builder is intended for evidence-downstream visualization.  It keeps
    the original shooter and ball qpos ordering, then attaches the passer body
    exactly as the coupled physics model does.  Stored trajectories therefore
    remain the source of every rendered pose; the derived stadium does not
    change or rescore their physics.
    """

    goal = spec or G1TrainingGoalSpec()
    root = asset_root.expanduser().resolve()
    parent = _stadium_spec(root, goal)

    import mujoco

    child = mujoco.MjSpec.from_file(str(root / _MODEL_REL))
    frame = parent.worldbody.add_frame(
        name="passer_frame",
        pos=passer_origin_m,
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    first_body = child.worldbody.first_body()
    if first_body is None:
        raise ValueError("qualified G1 model does not contain a root body")
    frame.attach_body(first_body, prefix="passer_")
    model = parent.compile()
    _require_stadium_model(model)
    if model.nu != 58:
        raise ValueError(f"coupled stadium model has {model.nu} actuators, expected 58")
    return model


def build_g1_three_player_stadium_model(
    asset_root: Path,
    *,
    passer_origin_m: tuple[float, float, float],
    goalkeeper_origin_m: tuple[float, float, float],
    spec: G1TrainingGoalSpec | None = None,
) -> Any:
    """Compile one shared pitch for passer, shooter, goalkeeper and ball.

    The unprefixed source body remains the shooter.  The two attached bodies
    use the same qualified G1 model and face back toward the shooter.  This is
    a physical three-body scene, not a video compositing helper.
    """

    goal = spec or G1TrainingGoalSpec()
    root = asset_root.expanduser().resolve()
    parent = _stadium_spec(root, goal)

    import mujoco

    _attach_g1(
        parent,
        root=root,
        frame_name="passer_frame",
        prefix="passer_",
        origin_m=passer_origin_m,
        yaw_rad=math.pi,
        mujoco=mujoco,
    )
    _attach_g1(
        parent,
        root=root,
        frame_name="goalkeeper_frame",
        prefix="goalkeeper_",
        origin_m=goalkeeper_origin_m,
        yaw_rad=math.pi,
        mujoco=mujoco,
    )
    model = parent.compile()
    _require_stadium_model(model)
    if model.nu != 87:
        raise ValueError(f"three-player stadium model has {model.nu} actuators, expected 87")
    return model


def _attach_g1(
    parent: Any,
    *,
    root: Path,
    frame_name: str,
    prefix: str,
    origin_m: tuple[float, float, float],
    yaw_rad: float,
    mujoco: Any,
) -> None:
    """Attach one qualified G1 without changing source ball/qpos ordering."""

    child = mujoco.MjSpec.from_file(str(root / _MODEL_REL))
    half_yaw = 0.5 * yaw_rad
    frame = parent.worldbody.add_frame(
        name=frame_name,
        pos=origin_m,
        quat=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
    )
    first_body = child.worldbody.first_body()
    if first_body is None:
        raise ValueError("qualified G1 model does not contain a root body")
    frame.attach_body(first_body, prefix=prefix)


def _stadium_spec(asset_root: Path, spec: G1TrainingGoalSpec) -> Any:
    import mujoco

    scene = asset_root.expanduser().resolve() / _SCENE_REL
    parent = mujoco.MjSpec.from_file(str(scene))
    wall = parent.body("box")
    if wall is None:
        raise ValueError("qualified G1 scene does not contain the replaceable box body")
    parent.delete(wall)
    _style_pitch_and_ball(parent, spec)
    _add_goal(parent, spec)
    return parent


def _require_stadium_model(model: Any) -> None:
    import mujoco

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box") >= 0:
        raise AssertionError("stadium scene retained the original wall")
    for name in ("goal_left_post", "goal_right_post", "goal_crossbar", "goal_back_net"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            raise AssertionError(f"stadium scene is missing {name}")


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


def g1_goal_net_contact_plane_x(
    spec: G1TrainingGoalSpec,
    *,
    capture_depth_m: float,
    ball_z_m: float,
) -> float:
    """Return the ball-centre contact plane aligned to the sloped visible net."""

    if not math.isfinite(capture_depth_m) or not 0.20 <= capture_depth_m <= spec.depth_m:
        raise ValueError("goal net capture depth must be inside the visible net")
    normalized_height = min(1.0, max(0.0, ball_z_m / spec.height_m))
    visible_depth = spec.depth_m * (1.0 - 0.32 * normalized_height)
    selected_depth = min(capture_depth_m, visible_depth)
    return spec.plane_x_m + selected_depth - spec.ball_radius_m


def apply_g1_compliant_goal_net_force(
    data: Any,
    *,
    ball_body_id: int,
    ball_qpos: int,
    ball_qvel: int,
    spec: G1TrainingGoalSpec,
    capture_depth_m: float,
    stiffness_n_m: float,
    damping_n_s_m: float,
) -> None:
    """Apply a bounded spring-damper force matching the visible back/side/roof net."""

    if not 10.0 <= stiffness_n_m <= 250.0:
        raise ValueError("goal net stiffness must be in [10, 250] N/m")
    if not 2.0 <= damping_n_s_m <= 30.0:
        raise ValueError("goal net damping must be in [2, 30] N s/m")
    data.xfrc_applied[ball_body_id, :] = 0.0
    x, y, z = (float(value) for value in data.qpos[ball_qpos : ball_qpos + 3])
    vx, vy, vz = (float(value) for value in data.qvel[ball_qvel : ball_qvel + 3])
    capture_x = g1_goal_net_contact_plane_x(
        spec,
        capture_depth_m=capture_depth_m,
        ball_z_m=z,
    )
    if x <= capture_x:
        if x > spec.plane_x_m and vx < 0.0:
            data.xfrc_applied[ball_body_id, :3] = (
                min(250.0, -damping_n_s_m * vx),
                max(-250.0, min(250.0, -0.12 * damping_n_s_m * vy)),
                max(-250.0, min(250.0, -0.08 * damping_n_s_m * vz)),
            )
        return

    penetration = x - capture_x
    engagement = min(1.0, max(0.0, penetration / 0.10))
    fx = -stiffness_n_m * penetration - engagement * damping_n_s_m * vx
    fy = -0.20 * engagement * damping_n_s_m * vy
    fz = -0.12 * engagement * damping_n_s_m * vz
    side_limit = spec.width_m / 2.0 - spec.ball_radius_m
    side_penetration = abs(y) - side_limit
    if x > spec.plane_x_m and side_penetration > 0.0:
        fy -= math.copysign(stiffness_n_m * side_penetration, y) + damping_n_s_m * vy
    roof_z = spec.height_m - spec.ball_radius_m
    if x > spec.plane_x_m and z > roof_z:
        fz -= stiffness_n_m * (z - roof_z) + damping_n_s_m * max(0.0, vz)
    data.xfrc_applied[ball_body_id, :3] = tuple(
        max(-250.0, min(250.0, value)) for value in (fx, fy, fz)
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
    # Thin half-metre visual mesh makes depth legible without creating any
    # collision wall. Ball retention comes only from the compliant force field.
    net = (0.91, 0.94, 0.92, 0.22)
    vertical_count = max(7, int(math.ceil(spec.width_m / 0.50)) + 1)
    horizontal_count = max(5, int(math.ceil(spec.height_m / 0.42)) + 1)
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
        side_horizontal_count = max(4, int(math.ceil(spec.height_m / 0.42)) + 1)
        for index in range(side_horizontal_count):
            z = spec.height_m * index / (side_horizontal_count - 1)
            rear_x = rear_bottom_x + (rear_top_x - rear_bottom_x) * z / spec.height_m
            capsule(
                f"goal_{side}_net_h_{index}",
                (x, y, max(0.025, z)),
                (rear_x, y, max(0.025, z)),
                rgba=net,
                collision=False,
                strand=True,
            )
        side_vertical_count = max(3, int(math.ceil(spec.depth_m / 0.50)) + 1)
        for index in range(side_vertical_count):
            alpha = index / (side_vertical_count - 1)
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
    for index in range(vertical_count):
        y = -half_width + spec.width_m * index / (vertical_count - 1)
        capsule(
            f"goal_roof_net_{index}",
            (x, y, spec.height_m),
            (rear_top_x, y, spec.height_m),
            rgba=net,
            collision=False,
            strand=True,
        )


def _style_pitch_and_ball(parent: Any, spec: G1TrainingGoalSpec) -> None:
    """Replace the blue grid with a pitch and add a lightweight ball pattern."""

    import mujoco

    floor = parent.geom("floor")
    floor.material = ""
    floor.rgba = (0.055, 0.24, 0.075, 1.0)
    world = parent.worldbody
    pitch_start_x = spec.plane_x_m - spec.field_length_m if spec.regulation_field_enabled else -5.0
    pitch_end_x = (
        spec.plane_x_m if spec.regulation_field_enabled else spec.plane_x_m + spec.depth_m + 1.0
    )
    pitch_half_width = spec.field_width_m / 2.0 if spec.regulation_field_enabled else 4.2
    stripe_count = 20 if spec.regulation_field_enabled else 10
    stripe_width = (pitch_end_x - pitch_start_x) / stripe_count
    for index in range(stripe_count):
        shade = 0.105 if index % 2 == 0 else 0.082
        world.add_geom(
            name=f"pitch_mowing_stripe_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(pitch_start_x + (index + 0.5) * stripe_width, 0.0, 0.0015),
            size=(stripe_width / 2.0, pitch_half_width, 0.0015),
            rgba=(0.035, shade + 0.12, 0.052, 1.0),
            contype=0,
            conaffinity=0,
        )
    line = (0.93, 0.94, 0.90, 0.92)
    line_half = spec.field_line_width_m / 2.0 if spec.regulation_field_enabled else 0.018
    box_depth = (
        spec.goal_area_depth_m
        if spec.regulation_field_enabled
        else min(2.2, max(1.4, spec.plane_x_m - 2.0))
    )
    box_half_width = (
        spec.width_m / 2.0 + spec.goal_area_depth_m if spec.regulation_field_enabled else 2.8
    )
    for name, pos, size in (
        (
            "pitch_goal_line",
            (spec.plane_x_m, 0.0, 0.004),
            (line_half, pitch_half_width, 0.003),
        ),
        (
            "pitch_box_front",
            (spec.plane_x_m - box_depth, 0.0, 0.004),
            (line_half, box_half_width, 0.003),
        ),
        (
            "pitch_box_left",
            (spec.plane_x_m - box_depth / 2.0, -box_half_width, 0.004),
            (box_depth / 2.0, line_half, 0.003),
        ),
        (
            "pitch_box_right",
            (spec.plane_x_m - box_depth / 2.0, box_half_width, 0.004),
            (box_depth / 2.0, line_half, 0.003),
        ),
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
    if spec.regulation_field_enabled:
        penalty_half_width = spec.width_m / 2.0 + spec.penalty_area_depth_m
        for name, pos, size in (
            (
                "pitch_penalty_front",
                (spec.plane_x_m - spec.penalty_area_depth_m, 0.0, 0.004),
                (line_half, penalty_half_width, 0.003),
            ),
            (
                "pitch_penalty_left",
                (
                    spec.plane_x_m - spec.penalty_area_depth_m / 2.0,
                    -penalty_half_width,
                    0.004,
                ),
                (spec.penalty_area_depth_m / 2.0, line_half, 0.003),
            ),
            (
                "pitch_penalty_right",
                (
                    spec.plane_x_m - spec.penalty_area_depth_m / 2.0,
                    penalty_half_width,
                    0.004,
                ),
                (spec.penalty_area_depth_m / 2.0, line_half, 0.003),
            ),
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
        world.add_geom(
            name="pitch_penalty_mark",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(spec.plane_x_m - spec.penalty_mark_distance_m, 0.0, 0.006),
            size=(0.11, 0.004, 0.0),
            rgba=line,
            contype=0,
            conaffinity=0,
        )
    ball = parent.body("ball")
    ball_geom = parent.geom("ball_geom")
    if ball_geom is None:
        raise ValueError("qualified stadium is missing ball_geom")
    ball_joints = list(ball.joints)
    if len(ball_joints) != 1 or ball_joints[0].name != "ball_free":
        raise ValueError("qualified stadium ball must expose exactly one ball_free joint")
    # The upstream demonstration scene uses 0.3 N*s/m on all six free-joint
    # DOFs.  For a 0.41 kg ball this erases about a quarter of the shot speed
    # per second and damps spin almost instantly, making flight and net entry
    # look submerged.  Keep only a small numerical damping term; goal capture
    # is handled separately by the compliant net after the back-net depth.
    ball_joints[0].damping = (spec.ball_free_joint_damping_n_s_m, 0.0, 0.0)
    radius = spec.ball_radius_m
    inertia = 0.4 * spec.ball_mass_kg * radius * radius
    ball.mass = spec.ball_mass_kg
    ball.inertia = (inertia, inertia, inertia)
    ball_geom.size = (radius, 0.0, 0.0)
    ball_geom.friction = (
        spec.ball_contact_sliding_friction,
        spec.ball_torsional_friction,
        spec.ball_rolling_friction,
    )
    ball_floor = parent.pair("ball_floor")
    if ball_floor is None:
        raise ValueError("qualified stadium is missing ball_floor contact pair")
    ball_floor.condim = 6
    ball_floor.friction = (
        spec.ball_sliding_friction,
        spec.ball_sliding_friction,
        spec.ball_torsional_friction,
        spec.ball_rolling_friction,
        spec.ball_rolling_friction,
    )
    patch_distance = radius - 0.003
    patch_radius = radius * 0.19
    patch_depth = radius * 0.025
    patches = (
        ((patch_distance, 0.0, 0.0), (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)),
        ((-patch_distance, 0.0, 0.0), (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)),
        ((0.0, patch_distance, 0.0), (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)),
        ((0.0, -patch_distance, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),
        ((0.0, 0.0, patch_distance), (1.0, 0.0, 0.0, 0.0)),
    )
    for index, (position, quaternion) in enumerate(patches):
        ball.add_geom(
            name=f"ball_patch_{index}",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=position,
            quat=quaternion,
            size=(patch_radius, patch_radius, patch_depth),
            rgba=(0.025, 0.025, 0.025, 1.0),
            density=0.0,
            contype=0,
            conaffinity=0,
        )


__all__ = [
    "G1TrainingGoalSpec",
    "apply_g1_compliant_goal_net_force",
    "build_g1_coupled_stadium_model",
    "build_g1_stadium_model",
    "build_g1_three_player_stadium_model",
    "g1_stadium_scene_hash",
    "g1_goal_net_contact_plane_x",
]
