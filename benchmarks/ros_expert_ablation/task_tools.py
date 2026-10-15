"""Read-only sealed task tools and restricted staging proposals, never ROS I/O."""

import copy
import json
import math
from datetime import datetime

import yaml

GENERIC = ("list_files", "read_file", "propose_patch", "select_interfaces", "run_tests", "finish")
EXPERT = ("ros_inspect_system", "ros_diagnose", "ros_context")


def parts(pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("RFC6901 JSON pointer required")
    return (
        [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")] if pointer else []
    )


def descend(value, pointer):
    for key in parts(pointer):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


class TaskTools:
    def __init__(self, files, *, arm, reviewed_at, minimum_radius_m, memory_query=None):
        if arm not in {"A", "B", "C"}:
            raise ValueError("closed experimental arm required")
        if (
            type(minimum_radius_m) not in (int, float)
            or not math.isfinite(minimum_radius_m)
            or minimum_radius_m <= 0
        ):
            raise ValueError("independently derived original URDF envelope required")
        self.minimum_radius_m = minimum_radius_m
        self.files = dict(files)
        self.arm = arm
        self.reviewed_at = reviewed_at
        self.memory_query = memory_query
        self.staging = {n: yaml.safe_load(self.files[n]) for n in ["nav2.yaml", "bringup.json"]}
        self.initial_staging = copy.deepcopy(self.staging)
        self.selected_interfaces = None
        self.model = None
        self.patch_trace = []
        self.memory_trace = []
        self.allowed = set()
        for namespace, group in self.staging["nav2.yaml"].items():
            if not isinstance(group, dict):
                continue
            for suffix in [
                ("amcl", "ros__parameters", "tf_broadcast"),
                ("lifecycle_manager_navigation", "ros__parameters", "node_names"),
                ("local_costmap", "local_costmap", "ros__parameters", "robot_radius"),
                ("global_costmap", "global_costmap", "ros__parameters", "robot_radius"),
            ]:
                self.allowed.add(("nav2.yaml", (namespace, *suffix)))
        self.allowed.update(
            {
                ("bringup.json", ("scan_bridge_state",)),
                ("bringup.json", ("additional_command_endpoint", "enabled")),
            }
        )

    @property
    def operations(self):
        return [
            *GENERIC,
            *(EXPERT if self.arm in {"B", "C"} else ()),
            *(("query_memory",) if self.arm == "C" else ()),
        ]

    def _model(self):
        if self.model is None:
            from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
            from rosclaw.connectors.ros.intelligence import build_system_model

            native = json.loads(self.files["native.json"])
            self.model = build_system_model(
                RosGraphSnapshot.from_dict(
                    {
                        "ros_version": "ros2",
                        "distro": "jazzy",
                        "endpoint": "fixture://sealed-original-task",
                        "captured_at": native["captured_at"],
                        **native["graph"],
                    }
                ),
                native=native,
            )
        return self.model

    def call(self, operation, args):
        if operation not in self.operations:
            raise ValueError("tool forbidden in this preregistered arm")
        if type(args) is not dict:
            raise ValueError("structured tool arguments required")
        if operation == "list_files":
            return {
                "files": sorted(self.files),
                "allowed_staging_fields": [
                    {
                        "file": file,
                        "pointer": "/"
                        + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path),
                    }
                    for file, path in sorted(self.allowed)
                ],
                "source_scope": "SEALED_ORIGINAL_NOT_CURRENT_L1",
            }
        if operation == "read_file":
            name = args["file"]
            if name not in self.files:
                raise ValueError("task file allowlist required")
            raw = self.files[name]
            if args.get("pointer"):
                value = json.loads(raw) if name.endswith(".json") else yaml.safe_load(raw)
                raw = json.dumps(descend(value, args["pointer"]), ensure_ascii=False)
            offset, limit = args.get("offset", 0), args.get("limit", 12000)
            if (
                type(offset) is not int
                or type(limit) is not int
                or offset < 0
                or not 1 <= limit <= 16000
            ):
                raise ValueError("bounded original file slice required")
            return {
                "file": name,
                "pointer": args.get("pointer", ""),
                "offset": offset,
                "total_characters": len(raw),
                "text": raw[offset : offset + limit],
                "truncated": offset + limit < len(raw),
            }
        if (
            self.arm == "C"
            and operation in {"propose_patch", "select_interfaces"}
            and not self.memory_trace
        ):
            raise ValueError("query provenance-bound training Memory before a proposal in arm C")
        if operation == "propose_patch":
            changes = args["changes"]
            if type(changes) is not list or not 1 <= len(changes) <= 8:
                raise ValueError("bounded staging patch required")
            proposed = copy.deepcopy(self.staging)
            trace = []
            for change in changes:
                name = change["file"]
                path = tuple(parts(change["pointer"]))
                value = change["value"]
                if (name, path) not in self.allowed:
                    raise ValueError("patch outside operator staging allowlist")
                leaf = path[-1]
                if leaf in {"enabled", "tf_broadcast"}:
                    valid = type(value) is bool
                elif leaf == "robot_radius":
                    valid = (
                        type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= self.minimum_radius_m
                    )
                elif leaf == "scan_bridge_state":
                    valid = value in {"running", "paused_after_initialization"}
                else:
                    valid = (
                        type(value) is list
                        and value
                        and all(type(v) is str for v in value)
                        and len(set(value)) == len(value)
                    )
                if not valid:
                    raise ValueError("typed staging field value required")
                parent = proposed[name]
                for key in path[:-1]:
                    parent = parent[key]
                prior = parent.get(leaf)
                parent[leaf] = copy.deepcopy(value)
                trace.append(
                    {"file": name, "pointer": change["pointer"], "before": prior, "after": value}
                )
            self.staging = proposed
            self.patch_trace.extend(trace)
            return {"status": "PROPOSED_STAGING_ONLY_NOT_PHYSICAL_ACCEPTANCE", "changes": trace}
        if operation == "select_interfaces":
            selected = args["interface_names"]
            if (
                type(selected) is not dict
                or set(selected)
                != {
                    "mapping.occupancy_map",
                    "state.odometry",
                    "sensing.lidar",
                    "navigation.navigate_to_pose",
                }
                or any(type(v) is not str or not v.startswith("/") for v in selected.values())
            ):
                raise ValueError(
                    "four explicit fully qualified observed interface selections required"
                )
            self.selected_interfaces = copy.deepcopy(selected)
            return {
                "status": "PROPOSED_SELECTION_REQUIRES_NEW_WORLD_BINDING",
                "interface_names": self.selected_interfaces,
            }
        if operation == "query_memory":
            if self.memory_query is None:
                raise ValueError("actual training Memory query unavailable")
            result = self.memory_query(args["query"])
            self.memory_trace.append({"query": args["query"], "retrieval_and_admission": result})
            return result
        if operation in EXPERT:
            model = self._model()
            if operation == "ros_inspect_system":
                return {
                    "schema_version": model.to_dict()["schema_version"],
                    "snapshot_hash": model.snapshot_hash,
                    "environment": model.environment,
                    "completeness": model.completeness,
                    "graph": model.graph,
                    "signals": model.to_dict()["signals"],
                    "lifecycle": model.to_dict()["lifecycle"],
                    "scope": "SEALED_ORIGINAL_ENGINEERING_SNAPSHOT_NOT_L1_AUTHORITY",
                }
            if operation == "ros_diagnose":
                from rosclaw.connectors.ros.diagnosis import diagnose

                return diagnose(
                    model,
                    profile="all",
                    now=datetime.fromisoformat(self.reviewed_at),
                    max_age_ms=500,
                )
            from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate

            req = json.loads(self.files["requirements.json"])
            if self.selected_interfaces is not None:
                req["interface_names"] = self.selected_interfaces
            return review_navigation_candidate(
                model,
                self.files["robot.urdf"].encode(),
                req,
                now=datetime.fromisoformat(self.reviewed_at),
            )
        raise ValueError("evaluation and finish are dispatched only by the external coordinator")
