"""Pinned operator inputs for the existing SDK witness; no robot profile.

Pure parsing only. Source matching and daemon authority remain separate gates.
"""

import hashlib
import re
from dataclasses import dataclass

from rosclaw.connectors.ros.context.geometry import _apply, _compose
from rosclaw.connectors.ros.context.navigation_workspace_review import _json
from rosclaw.connectors.ros.context.sim_source_binding import _pose, _transform, _xml, sdf_inventory

ROLES = {
    "observation",
    "lease",
    "ground_truth",
    "localization",
    "velocity_input",
    "drive_command",
    "collision_monitor_state",
}
NAME = re.compile(r"/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*")


def _ground_names(world, world_id, robot_entity):
    """Allow only exact scoped collision names of source-proven horizontal planes."""
    root = _xml(world)
    worlds = root.findall("world")
    if root.tag != "sdf" or len(worlds) != 1 or worlds[0].get("name") != world_id:
        raise ValueError("unique declared source World required")
    models = worlds[0].findall("model")
    if len([m for m in models if m.get("name") == robot_entity]) != 1:
        raise ValueError("unique declared source robot required")
    names = set()
    for model in models:
        if model.get("name") == robot_entity or model.findtext("static", "false") not in {
            "true",
            "1",
        }:
            continue
        inventory = sdf_inventory(
            world, robot_entity=model.get("name"), world_name=world_id, allow_plane=True
        )
        for link, collision, _, geometry in inventory["collisions"]:
            if geometry[0] != "plane":
                continue
            normal = dict(geometry[1])["normal"]
            transform = _compose(
                _transform(_pose(model.find("pose"))),
                inventory["member_transforms"][(link, collision)],
            )
            up = _apply(transform[0], normal)
            if abs(up[0]) > 1e-9 or abs(up[1]) > 1e-9 or up[2] <= 0:
                continue
            scoped = "::".join((model.get("name"), link, collision))
            names.update((scoped, world_id + "::" + scoped))
    return names


@dataclass(frozen=True)
class SimulationWitnessSources:
    run_id: str
    world_id: str
    robot_entity: str
    observer_node: str
    map_frame: str
    source_urdf_sha256: str
    world_sdf_sha256: str
    topics: tuple[tuple[str, str], ...]
    contact_topics: tuple[str, ...]
    ground_witness_topics: tuple[str, ...]
    ground_collision_names: frozenset[str]
    drive_twist_stamped: bool

    def topic(self, role):
        return dict(self.topics)[role]

    @classmethod
    def from_bytes(cls, raw, *, expected_sha256, urdf_bytes, world_sdf_bytes):
        if (
            type(raw) is not bytes
            or not 0 < len(raw) <= 64_000
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise ValueError("independently pinned bounded observer inputs required")
        data = _json(raw)
        keys = {
            "schema_version",
            "run_id",
            "world_id",
            "robot_entity",
            "observer_node",
            "map_frame",
            "source_urdf_sha256",
            "world_sdf_sha256",
            "topics",
            "contact_topics",
            "ground_witness_topics",
            "ground_collision_names",
            "drive_twist_stamped",
        }
        if (
            type(data) is not dict
            or set(data) != keys
            or data["schema_version"] != "rosclaw.simulation_witness_sources.v1"
        ):
            raise ValueError("closed operator observer source contract required")
        if type(data["run_id"]) is not str or not re.fullmatch(r"[a-f0-9]{32}", data["run_id"]):
            raise ValueError("explicit fresh SIM run identity required")
        if any(
            type(data[k]) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", data[k])
            for k in ("world_id", "robot_entity")
        ):
            raise ValueError("explicit resolved World and model names required")
        if (
            type(data["map_frame"]) is not str
            or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9/]*", data["map_frame"])
            or "//" in data["map_frame"]
            or data["map_frame"].endswith("/")
        ):
            raise ValueError("explicit canonical map frame required")
        if (
            type(data["topics"]) is not dict
            or set(data["topics"]) != ROLES
            or type(data["drive_twist_stamped"]) is not bool
        ):
            raise ValueError("explicit complete typed witness routes required")
        values = [data["observer_node"], *data["topics"].values()]
        if any(type(n) is not str or not NAME.fullmatch(n) for n in values) or len(
            set(values)
        ) != len(values):
            raise ValueError("distinct fully qualified observer routes required")
        for key in ("contact_topics", "ground_witness_topics", "ground_collision_names"):
            rows = data[key]
            if (
                type(rows) is not list
                or not 1 <= len(rows) <= 64
                or any(type(n) is not str or not n or len(n) > 512 for n in rows)
                or len(set(rows)) != len(rows)
            ):
                raise ValueError("bounded explicit unique contact sources required")
        if any(not NAME.fullmatch(n) or n in values for n in data["contact_topics"]) or not set(
            data["ground_witness_topics"]
        ) <= set(data["contact_topics"]):
            raise ValueError("ground witnesses must be explicit observed contact topics")
        if (
            data["source_urdf_sha256"] != hashlib.sha256(urdf_bytes).hexdigest()
            or data["world_sdf_sha256"] != hashlib.sha256(world_sdf_bytes).hexdigest()
        ):
            raise ValueError("observer source assets differ from independent pins")
        if not set(data["ground_collision_names"]) <= _ground_names(
            world_sdf_bytes, data["world_id"], data["robot_entity"]
        ):
            raise ValueError("ground exemptions must name actual static horizontal planes")
        return cls(
            **{
                k: data[k]
                for k in keys
                - {
                    "schema_version",
                    "topics",
                    "contact_topics",
                    "ground_witness_topics",
                    "ground_collision_names",
                }
            },
            topics=tuple(sorted(data["topics"].items())),
            contact_topics=tuple(data["contact_topics"]),
            ground_witness_topics=tuple(data["ground_witness_topics"]),
            ground_collision_names=frozenset(data["ground_collision_names"]),
        )
