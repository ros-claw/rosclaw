"""Read-only spatial source constraint for an isolated SIM instrument.

This matches original scene geometry to the independent native source entity
inventory. It grants no scene-service or robot authority, and cannot certify
world ownership, DDS authentication, physical stopping or task acceptance.
"""

import math
from copy import deepcopy

from backend_probe_fixture import validate_probe_declaration
from native_contact_evidence import decode_native_packet

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.verification.occupancy_geometry import parse_physics_packet


class ProbeSceneGeometry:
    def __init__(self, binding, declaration, joint_gate):
        d = validate_probe_declaration(declaration)
        robot, probe = joint_gate.robot.policy, joint_gate.probe.policy["native_policy"]
        r, p = robot["contact_policy"], probe["contact_policy"]
        if (
            binding["run_id"] != d["run_id"]
            or binding["run_id"] != r["run_id"]
            or binding["world_name"] != d["world_name"]
            or binding["world_name"] != robot["world_name"]
            or binding["body_model_name"] != d["robot_model_name"]
            or binding["body_model_name"] != r["model_name"]
            or d["probe_model_name"] != p["model_name"]
            or d["probe_model_name"] not in binding["obstacle_names"]
            or binding["body_snapshot_hash"] != r["body_snapshot_hash"]
            or binding["attachment_hash"] != r["attachment_hash"]
            or binding["grid"]["cleaning_polygon"] != d["cleaning_polygon"]
            or type(binding.get("world_to_map_xyyaw")) is not list
            or any(type(v) not in (int, float) for v in binding.get("world_to_map_xyyaw", []))
            or binding.get("world_to_map_xyyaw") != [0, 0, 0]
            or binding.get("map_world_identity_approved") is not True
            or binding.get("frame_transform_source") != "simulator_operator_fixture_policy"
        ):
            raise ValueError("same frozen owned scene/run/Body/region source roles required")
        self.binding, self.declaration, self.gate = deepcopy(binding), deepcopy(d), joint_gate
        self.last = self.geometry_hash = None
        self.fault = None
        self.opened = False
        self.actual_body_radius = None
        self.policy_hash = digest(
            {"binding": binding, "declaration": d, "joint_policy_hash": joint_gate.policy_hash}
        )

    def observe(
        self,
        raw,
        *,
        received_monotonic_sec,
        received_unix_ns,
        robot_component_bytes,
        probe_component_bytes,
    ):
        try:
            if self.fault:
                raise ValueError("instrument scene geometry rejection remains latched")
            wall = received_monotonic_sec
            if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
                raise ValueError("finite original scene receipt clock required")
            b, d = self.binding, self.declaration
            parsed = parse_physics_packet(
                raw,
                **{
                    k: b[k]
                    for k in (
                        "run_id",
                        "body_snapshot_hash",
                        "attachment_hash",
                        "world_name",
                        "body_model_name",
                    )
                },
                obstacle_names=tuple(b["obstacle_names"]),
                scene_model_names=frozenset(b["scene_model_names"]),
                received_at_unix_ns=received_unix_ns,
                maximum_body_planar_radius_m=d["maximum_robot_radius_m"],
            )
            packet = parsed["packet"]
            if packet["paused"]:
                raise ValueError("paused actual scene cannot qualify instrument placement")
            current = (
                packet["sequence"],
                packet["physics_iteration"],
                packet["sim_time_sec"],
                packet["captured_at_unix_ns"],
                wall,
            )
            if self.last and (
                any(a <= c for a, c in zip(current, self.last, strict=True))
                or current[0] != self.last[0] + 1
            ):
                raise ValueError(
                    "scene geometry original sequence/iteration/clocks missing or regressed"
                )
            robot, _ = decode_native_packet(robot_component_bytes, self.gate.robot.policy)
            probe, _ = decode_native_packet(
                probe_component_bytes, self.gate.probe.policy["native_policy"]
            )
            if robot["world_entity_id"] != probe["world_entity_id"]:
                raise ValueError("independent actual robot/probe world identities differ")
            for native, model in (
                (robot, packet["body"]),
                (
                    probe,
                    next(
                        o for o in packet["obstacles"] if o["model_name"] == d["probe_model_name"]
                    ),
                ),
            ):
                if (
                    native["body_model_entity_id"] != model["entity_id"]
                    or abs(native["sim_time_sec"] - packet["sim_time_sec"]) > 1e-9
                    or native["iterations"] != packet["physics_iteration"]
                ):
                    raise ValueError(
                        "exact original same-step scene/native model identity and clock required"
                    )
                if not 0 <= received_unix_ns - native["captured_at_unix_ns"] < 300_000_000:
                    raise ValueError("original native counterpart stale or future")
                native_ids = {row["collision_entity_id"] for row in native["collision_contacts"]}
                geometry_ids = {row["entity_id"] for row in model["collision_geometry"]}
                if (
                    native_ids != geometry_ids
                    or math.dist(native["body_world_pose"][:3], model["world_pose"][:3]) > 0.001
                ):
                    raise ValueError(
                        "complete actual collision identities or same-step physical poses differ"
                    )
                dot = abs(
                    sum(
                        a * c
                        for a, c in zip(
                            native["body_world_pose"][3:], model["world_pose"][3:], strict=True
                        )
                    )
                )
                if dot < math.cos(0.001 / 2):
                    raise ValueError("same-step original scene/native model orientations differ")
            instrument = next(
                o for o in packet["obstacles"] if o["model_name"] == d["probe_model_name"]
            )
            shapes = instrument["collision_geometry"]
            if (
                len(shapes) != 1
                or shapes[0]["kind"] != "sphere"
                or not math.isclose(
                    shapes[0]["radius"], d["sphere_radius_m"], rel_tol=0, abs_tol=1e-12
                )
                or shapes[0]["model_relative_pose"] != [0, 0, 0, 1, 0, 0, 0]
            ):
                raise ValueError(
                    "one actual centered instrument sphere must match sealed declaration"
                )
            position = instrument["world_pose"][:2]
            if math.dist(position, d["probe_xy"]) > 0.1:
                raise ValueError("actual instrument left its isolated declared column")
            radius = parsed["body_planar_radius_m"]
            if (
                math.dist(packet["body"]["world_pose"][:2], position)
                <= radius + d["sphere_radius_m"] + 0.5
            ):
                raise ValueError("actual robot collision envelope lacks instrument clearance")
            for obstacle in packet["obstacles"]:
                if obstacle["model_name"] == d["probe_model_name"]:
                    continue
                extent = dict(parsed["geometry"].model_radii)[obstacle["model_name"]]
                if (
                    math.dist(obstacle["world_pose"][:2], position)
                    <= extent + d["sphere_radius_m"] + 0.2
                ):
                    raise ValueError("actual scene obstacle approaches instrument column")
            geometry_hash = digest(
                {
                    "world_entity_id": robot["world_entity_id"],
                    "scene_models": sorted(
                        packet["scene_models"], key=lambda model: model["model_name"]
                    ),
                    "body": [
                        {
                            k: value
                            for k, value in shape.items()
                            if k not in {"model_relative_pose", "enclosing_radius_m"}
                        }
                        for shape in packet["body"]["collision_geometry"]
                    ],
                    "obstacles": sorted(
                        [
                            {
                                "model_name": o["model_name"],
                                "entity_id": o["entity_id"],
                                "collision_geometry": o["collision_geometry"],
                            }
                            for o in packet["obstacles"]
                        ],
                        key=lambda o: o["model_name"],
                    ),
                }
            )
            if self.geometry_hash is not None and self.geometry_hash != geometry_hash:
                raise ValueError("sealed actual collision geometry inventory changed")
            self.last, self.geometry_hash, self.actual_body_radius = current, geometry_hash, radius
            return self.snapshot(wall)
        except (ValueError, TypeError, KeyError, StopIteration) as exc:
            self.fault = self.fault or str(exc)
            raise ValueError(f"original instrument scene geometry rejected: {exc}") from exc

    def snapshot(self, wall):
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            self.fault = self.fault or "invalid instrument geometry sample clock"
            raise ValueError("finite scene geometry sample clock required")
        fresh = self.last is not None and 0 <= wall - self.last[4] < 0.3
        if self.opened and not fresh:
            self.fault = self.fault or "actual scene geometry expired after spatial admission"
        ready = fresh and self.fault is None
        self.opened = self.opened or ready
        return {
            "evidence_role": "additional_actual_scene_geometry_correspondence_not_permission",
            "scene_geometry_constraint_satisfied": ready,
            "policy_hash": self.policy_hash,
            "actual_body_planar_radius_m": self.actual_body_radius,
            "actual_geometry_hash": self.geometry_hash,
            "source_fault": self.fault,
            "world_source_ownership_admitted": False,
            "runtime_controller_integration": "NOT_JOINED",
            "backend_health_admitted": False,
            "physical_acceptance": "NOT_VERIFIED",
            "authorization": False,
        }
