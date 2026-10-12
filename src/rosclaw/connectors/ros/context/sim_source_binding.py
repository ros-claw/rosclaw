"""SIM source capture review, with separate operator pins and no authority.

The operator must pin launch AND capture independently of the checked artifact.
A fabricated capture accompanied by a fabricated pin is outside this trust
assumption. Same-UID fixture supervision is not adversarial authentication.
"""

import hashlib
import math
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from rosclaw.body.schema import EffectiveBody
from rosclaw.connectors.ros.context.geometry import IDENTITY, ZERO, _compose, _inverse, _origin
from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.context.navigation_workspace_review import _inspect, _json, _Reader
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.contracts.common import content_hash


def _sha(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 30_000_000:
        raise ValueError("bounded captured bytes required")
    return hashlib.sha256(raw).hexdigest()


def _xml(raw):
    if len(raw) > 5_000_000 or any(x in raw for x in (b"<!DOCTYPE", b"<!ENTITY", b"${", b"$(")):
        raise ValueError("bounded resolved XML required")
    return ET.fromstring(raw)


def _numbers(text, count, *, positive=False):
    values = tuple(float(x) for x in text.split())
    if len(values) != count or any(not math.isfinite(x) for x in values):
        raise ValueError("finite geometry coordinates required")
    if positive and min(values) <= 0:
        raise ValueError("positive geometry dimensions required")
    # Subnanometre serialization normalization, never a task tolerance.
    normalized = tuple(round(x, 9) + 0.0 for x in values)
    if positive and min(normalized) <= 0:
        raise ValueError("geometry below supported serialization precision")
    return normalized


def _transform(pose):
    element = ET.Element("frame")
    ET.SubElement(
        element, "origin", xyz=" ".join(map(str, pose[:3])), rpy=" ".join(map(str, pose[3:]))
    )
    return _origin(element)


def _pose_matches(actual, expected, position_tolerance, rotation_tolerance):
    ar, ap = actual
    er, ep = expected
    return (
        max(abs(a - b) for a, b in zip(ap, ep, strict=True)) <= position_tolerance
        and max(abs(ar[i][j] - er[i][j]) for i in range(3) for j in range(3)) <= rotation_tolerance
    )


def _pose(element):
    if element is None:
        return (0.0,) * 6
    if element.attrib:
        raise ValueError("unresolved collision/sensor relative pose")
    return _numbers(element.text or "", 6)


def _geometry(element):
    if element is None or len(element) != 1:
        raise ValueError("one resolved collision primitive required")
    shape = element[0]
    fields = {
        "box": (("size", 3),),
        "sphere": (("radius", 1),),
        "cylinder": (("radius", 1), ("length", 1)),
    }
    if shape.tag not in fields:
        raise ValueError("unsupported collision geometry: " + shape.tag)
    return (
        shape.tag,
        tuple(
            (name, _numbers(shape.findtext(name, ""), count, positive=True))
            for name, count in fields[shape.tag]
        ),
    )


def sdf_inventory(raw, *, robot_entity, world_name=None):
    """Read expanded SDF, preserving link/joint structure and all collisions."""
    root = _xml(raw)
    if root.tag != "sdf" or root.findall(".//include"):
        raise ValueError("expanded SDF with resolved frames required")
    if world_name is None:
        models = root.findall("model")
    else:
        worlds = root.findall("world")
        if len(worlds) != 1 or worlds[0].get("name") != world_name:
            raise ValueError("explicit unique source World required")
        models = [x for x in worlds[0].findall("model") if x.get("name") == robot_entity]
    if len(models) != 1 or models[0].get("name") != robot_entity or models[0].findall("model"):
        raise ValueError("explicit unique nonnested source model required")
    model = models[0]
    nodes = model.findall("link") + model.findall("joint") + model.findall("frame")
    references = {"__model__": None}
    for node in nodes:
        name = node.get("name")
        if not name or name in references:
            raise ValueError("unique named SDF frame graph required")
        pose = node.find("pose")
        if pose is not None:
            if set(pose.attrib) - {"relative_to"}:
                raise ValueError("unsupported SDF pose representation")
            _numbers(pose.text or "", 6)
        default = (
            node.get("attached_to", "__model__")
            if node.tag == "frame"
            else node.findtext("child", "__model__")
            if node.tag == "joint"
            else "__model__"
        )
        references[name] = pose.get("relative_to", default) if pose is not None else default
    for name in references:
        seen, current = set(), name
        while current != "__model__":
            if current in seen or current not in references:
                raise ValueError("cyclic or unresolved SDF frame graph")
            seen.add(current)
            current = references[current]
    for node in model.findall("frame"):
        if node.get("attached_to", "__model__") not in references:
            raise ValueError("unresolved SDF frame attachment")
    transforms = {"__model__": (IDENTITY, ZERO)}
    local = {
        node.get("name"): _transform(_numbers(node.findtext("pose", "0 0 0 0 0 0"), 6))
        for node in nodes
    }

    def resolve(name):
        if name not in transforms:
            transforms[name] = _compose(resolve(references[name]), local[name])
        return transforms[name]

    for name in references:
        resolve(name)
    collisions, sensors, structure, links = [], [], [], set()
    model = models[0]
    for link in model.findall("link"):
        name = link.get("name")
        if not name or name in links:
            raise ValueError("unique named source links required")
        links.add(name)
        structure.append(
            (
                name,
                tuple(
                    ET.canonicalize(ET.tostring(x, encoding="unicode"), strip_text=True)
                    for x in link
                    if x.tag in {"pose", "inertial"}
                ),
            )
        )
        for member in link.findall("collision"):
            collisions.append(
                (
                    name,
                    member.get("name"),
                    _pose(member.find("pose")),
                    _geometry(member.find("geometry")),
                )
            )
        for member in link.findall("sensor"):
            sensors.append(
                (name, member.get("name"), _pose(member.find("pose")), member.get("type"))
            )
    joints = tuple(
        sorted(
            ET.canonicalize(ET.tostring(x, encoding="unicode"), strip_text=True)
            for x in model.findall("joint") + model.findall("frame")
        )
    )
    if not collisions or len(collisions) > 4096 or len(sensors) > 4096:
        raise ValueError("bounded nonempty source collisions required")
    for rows in (collisions, sensors):
        if any(not x[1] for x in rows) or len({x[:2] for x in rows}) != len(rows):
            raise ValueError("unique named source members required")
    return {
        "collisions": sorted(collisions),
        "sensors": sorted(sensors),
        "structure": (sorted(structure), joints),
        "member_transforms": {
            (name, member.get("name")): _compose(
                transforms[name], _transform(_pose(member.find("pose")))
            )
            for link in model.findall("link")
            for name in [link.get("name")]
            for member in [*link.findall("collision"), *link.findall("sensor")]
        },
    }


def _loaded_inventory(sample, entity):
    result = {"collisions": [], "sensors": []}
    for field in result:
        for row in sample[field]:
            ancestry = row["ancestry"]
            if not any(x["entity"] == entity for x in ancestry):
                continue
            if (
                not ancestry
                or ancestry[0].get("model")
                or ancestry[0].get("world")
                or sum(x.get("model") is True for x in ancestry) != 1
            ):
                raise ValueError("explicit nonnested member parent link required")
            pose = _numbers(" ".join(str(x) for x in row["pose"]), 6)
            value = (
                _geometry(_xml(row["geometry_sdf"].encode()))
                if field == "collisions"
                else row["kind"]
            )
            if value == "UNKNOWN":
                raise ValueError("unresolved loaded sensor type")
            result[field].append((ancestry[0]["name"], row["name"], pose, value))
        if len({x[:2] for x in result[field]}) != len(result[field]):
            raise ValueError("duplicate loaded member identity")
        result[field].sort()
    return result


def review_sim_source_binding(
    workspace,
    request,
    *,
    launch_bytes,
    capture_bytes,
    converted_sdf_bytes,
    world_sdf_bytes,
    ecm_bytes,
    model,
    now=None,
):
    """Review checked supervisor capture, not self-authenticating publisher JSON.

    MATCHED means correspondence at capture time under the external operator
    trust pins; it never issues a Permit, changes a Body, or proves L1/L2.
    """
    now = now or datetime.now(UTC)
    keys = {
        "schema_version",
        "manifest_hash",
        "body_snapshot_hash",
        "launch_sha256",
        "capture_sha256",
        "source_asset_ref",
        "source_asset_digest",
        "sim_world_id",
        "sim_run_id",
        "expected_robot_entity",
        "expected_ros_domain",
        "expected_namespace",
        "expected_frames",
        "max_age_ms",
    }
    if (
        type(request) is not dict
        or set(request) != keys
        or request["schema_version"] != "rosclaw.sim_source_binding_request.v1"
    ):
        raise ValueError("closed explicit source-binding request required")
    age = request["max_age_ms"]
    if type(age) not in (int, float) or not math.isfinite(age) or not 0 < age <= 10_000:
        raise ValueError("bounded explicit capture freshness policy required")
    if (
        type(request["expected_ros_domain"]) is not int
        or not 0 <= request["expected_ros_domain"] <= 232
    ):
        raise ValueError("explicit portable ROS domain required")
    if type(request["expected_namespace"]) is not str or not request[
        "expected_namespace"
    ].startswith("/"):
        raise ValueError("explicit absolute namespace required")
    for key in keys - {"schema_version", "max_age_ms", "expected_ros_domain", "expected_frames"}:
        if type(request[key]) is not str or not request[key] or len(request[key]) > 512:
            raise ValueError("explicit bounded source identities required")
    mismatch, unknown = [], []
    evidence = {
        "schema_version": "rosclaw.sim_source_binding_evidence.v1",
        "result": "UNKNOWN",
        "status": "SIM_SOURCE_UNKNOWN",
        "manifest_hash": request["manifest_hash"],
        "body_hash": request["body_snapshot_hash"],
        "world_id": request["sim_world_id"],
        "sim_run_id": request["sim_run_id"],
        "model_entity_id": None,
        "loaded_geometry_fingerprint": None,
        "ros_runtime_snapshot_hash": model.snapshot_hash,
        "observed_at": None,
        "reviewed_at": now.isoformat(),
        "source_provenance_class": "NOT_VERIFIED",
        "execution_authority": "NONE",
        "real_authority": False,
        "L1": "NOT_RUN",
        "L2": "NOT_RUN",
        "workspace_modified": False,
        "evidence_refs": [],
    }
    try:
        if (
            _sha(launch_bytes) != request["launch_sha256"]
            or _sha(capture_bytes) != request["capture_sha256"]
        ):
            mismatch.append("INDEPENDENT_OPERATOR_PIN_MISMATCH")
            raise ValueError("do not consume unpinned source claims")
        launch, capture = _json(launch_bytes), _json(capture_bytes)
        evidence["evidence_refs"] = [
            {"role": role, "sha256": _sha(raw)}
            for role, raw in (
                ("operator_launch", launch_bytes),
                ("operator_capture", capture_bytes),
                ("converted_sdf", converted_sdf_bytes),
                ("world_sdf", world_sdf_bytes),
                ("original_ecm", ecm_bytes),
            )
        ]
        if (
            launch["schema_version"] != "rosclaw.sim_source_launch.v1"
            or capture["schema_version"] != "rosclaw.sim_source_capture.v1"
        ):
            raise ValueError("source launch/capture schema required")
        for key, wanted in (
            ("world_id", request["sim_world_id"]),
            ("run_id", request["sim_run_id"]),
            ("robot_entity", request["expected_robot_entity"]),
            ("ros_domain", request["expected_ros_domain"]),
            ("namespace", request["expected_namespace"]),
        ):
            if launch[key] != wanted or capture[key] != wanted:
                mismatch.append("SOURCE_IDENTITY_MISMATCH:" + key)
        if (
            capture["launch_sha256"] != request["launch_sha256"]
            or capture["ecm_sha256"] != _sha(ecm_bytes)
            or capture["ros_snapshot_hash"] != model.snapshot_hash
        ):
            mismatch.append("CAPTURE_INPUT_HASH_MISMATCH")
        if (
            capture["assets_before"] != launch["asset_sha256"]
            or capture["assets_after"] != launch["asset_sha256"]
        ):
            mismatch.append("SOURCE_ASSET_REPLACED")
        assets = launch["asset_sha256"]
        if (
            assets["converted.sdf"] != _sha(converted_sdf_bytes)
            or assets["world.sdf"] != _sha(world_sdf_bytes)
            or assets[request["source_asset_ref"]] != request["source_asset_digest"]
        ):
            mismatch.append("SOURCE_ASSET_PIN_MISMATCH")
        conversion = launch["conversion"]
        if (
            conversion["argv"] != ["gz", "sdf", "-p", "/assets/robot.urdf"]
            or conversion["exit_code"] != 0
            or conversion["output_sha256"] != _sha(converted_sdf_bytes)
            or conversion["input_sha256"] != assets["robot.urdf"]
        ):
            unknown.append("UNVERIFIED_URDF_TO_SDF_LINEAGE")
        stamp = datetime.fromisoformat(capture["observed_at"])
        evidence["observed_at"] = stamp.isoformat()
        if (
            not 0 <= (now - stamp).total_seconds() * 1000 <= age
            or not 0 <= model.age_ms(now) <= age
        ):
            unknown.append("STALE_OR_FUTURE_CAPTURE")
        before, after = capture["container_before"], capture["container_after"]
        if not (
            before["Id"] == after["Id"]
            and before["Image"] == after["Image"] == launch["image_digest"]
            and before["State"]["Running"] is True
            and after["State"]["Running"] is True
            and before["State"]["Pid"] == after["State"]["Pid"] > 0
        ):
            unknown.append("UNVERIFIED_RUNNING_INSTANCE")
        for state in (before, after):
            if (
                state["HostConfig"]["ReadonlyRootfs"] is not True
                or state["HostConfig"]["NetworkMode"] != "none"
            ):
                unknown.append("UNISOLATED_SOURCE_INSTANCE")
            mounts = state["Mounts"]
            if not {"/assets", "/observer"} <= {
                x["Destination"] for x in mounts if x["RW"] is False
            } or any(x["RW"] for x in mounts if x["Destination"] in {"/assets", "/observer"}):
                unknown.append("SOURCE_OR_OBSERVER_NOT_READONLY")
        observer, witness = capture["observer"], capture["physics_witness"]
        if (
            observer["library_sha256"] != launch["observer_library_sha256"]
            or observer["server_count"] != 1
            or observer["alive"] is not True
        ):
            unknown.append("OBSERVER_PROVENANCE_UNVERIFIED")
        if (
            witness["identity"] != launch["physics_witness_identity"]
            or witness["alive"] is not True
            or witness["container_id"] != before["Id"]
            or witness["run_id"] != request["sim_run_id"]
            or witness["body_hash"] != request["body_snapshot_hash"]
        ):
            unknown.append("PHYSICS_WITNESS_PROVENANCE_UNVERIFIED")
        if witness["implementation_sha256"] != launch["physics_witness_implementation_sha256"]:
            unknown.append("PHYSICS_WITNESS_IMPLEMENTATION_MISMATCH")
        if str(model.environment.get("domain_id")) != str(request["expected_ros_domain"]):
            mismatch.append("ROS_DOMAIN_MISMATCH")
        with _Reader(workspace) as reader:
            files = reader.read()
            inspected = _inspect(
                files, request["manifest_hash"], request["body_snapshot_hash"], now
            )
            binding = SimulationNavigationBinding.from_body(
                EffectiveBody.from_dict(_json(files["body/refs/effective_body.json"]))
            )
            if (
                binding.run_id != request["sim_run_id"]
                or binding.observation_topic != launch["physics_witness_identity"]
            ):
                mismatch.append("BODY_SIM_RUN_OR_WITNESS_MISMATCH")
            if inspected["source_urdf_sha256"] != assets["robot.urdf"]:
                mismatch.append("BODY_URDF_SOURCE_MISMATCH")
            review = review_navigation_candidate(
                model,
                files["body/refs/robot.urdf"],
                _json(files["navigation-requirements.json"]),
                now=now,
            )
            if review["status"] != "PROPOSED":
                unknown.extend("ROS_RUNTIME:" + x for x in review["unknown_fields"])
            if review["candidate"]["frames"] != request["expected_frames"]:
                mismatch.append("BODY_ROS_FRAME_MISMATCH")
            for rows in review["candidate"]["interfaces"].values():
                for row in rows:
                    if request["expected_namespace"] != "/" and not row["name"].startswith(
                        request["expected_namespace"].rstrip("/") + "/"
                    ):
                        mismatch.append("ROS_NAMESPACE_MISMATCH")
            converted = sdf_inventory(
                converted_sdf_bytes, robot_entity=request["expected_robot_entity"]
            )
            expected = sdf_inventory(
                world_sdf_bytes,
                robot_entity=request["expected_robot_entity"],
                world_name=request["sim_world_id"],
            )
            position_tolerance = launch["geometry_pose_tolerance_m"]
            rotation_tolerance = launch["geometry_rotation_matrix_tolerance"]
            if (
                type(position_tolerance) not in (int, float)
                or not 0 < position_tolerance <= 0.0001
                or type(rotation_tolerance) not in (int, float)
                or not 0 < rotation_tolerance <= 0.001
            ):
                raise ValueError(
                    "bounded explicit stationary geometry comparison precision required"
                )
            world_models = _xml(world_sdf_bytes).find("world").findall("model")
            expected_models = [x.get("name") for x in world_models]
            if len(set(expected_models)) != len(expected_models) or any(
                x.findall("model") or not x.get("name") for x in world_models
            ):
                raise ValueError("unique nonnested complete World model inventory required")
            if (
                converted["collisions"] != expected["collisions"]
                or converted["structure"] != expected["structure"]
            ):
                mismatch.append("WORLD_CHANGED_CONVERTED_BODY_GEOMETRY_OR_STRUCTURE")
            samples = [_json(x) for x in ecm_bytes.splitlines() if x.strip()]
            if not 2 <= len(samples) <= 600:
                raise ValueError("bounded advancing ECM capture required")
            entity, previous = None, None
            for sample in samples:
                if (
                    sample["schema_version"] != "rosclaw.gazebo_ecm_source_sample.v1"
                    or sample["inventory_complete"] is not True
                    or sample["paused"] is not False
                    or sample["reset_epoch"] != 0
                ):
                    unknown.append("INCOMPLETE_PAUSED_OR_RESET_WORLD")
                if sample["observer_run_label"] != request["sim_run_id"]:
                    mismatch.append("OBSERVER_RUN_MISMATCH")
                worlds = sample["worlds"]
                if sorted(x["name"] for x in sample["models"]) != sorted(expected_models):
                    mismatch.append("WORLD_MODEL_INVENTORY_MISMATCH")
                models = [
                    x for x in sample["models"] if x["name"] == request["expected_robot_entity"]
                ]
                if (
                    len(worlds) != 1
                    or worlds[0]["name"] != request["sim_world_id"]
                    or len(models) != 1
                ):
                    mismatch.append("WORLD_OR_MODEL_NOT_UNIQUE")
                    continue
                current = models[0]["entity"]
                if entity is not None and current != entity:
                    unknown.append("MODEL_ENTITY_REPLACED")
                entity = current
                if (
                    witness["entity_id"] != entity
                    or witness["world_entity_id"] != worlds[0]["entity"]
                ):
                    mismatch.append("WITNESS_WORLD_ENTITY_MISMATCH")
                if previous and any(
                    sample[k] <= previous[k]
                    for k in ("sim_time_ns", "iterations", "observer_monotonic_ns", "sample")
                ):
                    unknown.append("CLOCK_OR_CAPTURE_SEQUENCE_DISCONTINUITY")
                previous = sample
                loaded = _loaded_inventory(sample, entity)
                model_inverse = _inverse(
                    _transform(_numbers(" ".join(map(str, models[0]["world_pose"])), 6))
                )
                for field in ("collisions", "sensors"):
                    for row in sample[field]:
                        if not any(x["entity"] == entity for x in row["ancestry"]):
                            continue
                        name = (row["ancestry"][0]["name"], row["name"])
                        actual_pose = _compose(
                            model_inverse,
                            _transform(_numbers(" ".join(map(str, row["world_pose"])), 6)),
                        )
                        if name not in expected["member_transforms"] or not _pose_matches(
                            actual_pose,
                            expected["member_transforms"][name],
                            position_tolerance,
                            rotation_tolerance,
                        ):
                            mismatch.append("LOADED_MEMBER_EFFECTIVE_FRAME_MISMATCH")
                if (
                    loaded["collisions"] != expected["collisions"]
                    or loaded["sensors"] != expected["sensors"]
                ):
                    mismatch.append("LOADED_GEOMETRY_OR_SENSOR_MOUNT_MISMATCH")
                evidence["loaded_geometry_fingerprint"] = content_hash("simgeometry", loaded)
            evidence["model_entity_id"] = entity
            physics = witness["sample"]
            physics_stamp = datetime.fromisoformat(physics["captured_at"])
            if (
                not 0 <= (now - physics_stamp).total_seconds() * 1000 <= age
                or physics["observation_complete"] is not True
                or physics["source_only_observer"] is not True
                or physics["lease_enabled"] is not False
                or physics["source_received_control_messages"] != 0
                or physics["source_published_velocity_messages"] != 0
            ):
                unknown.append("PHYSICS_SOURCE_ONLY_OBSERVATION_UNVERIFIED")
            if (
                physics["body_snapshot_hash"] != binding.body_snapshot_hash
                or physics["run_id"] != binding.run_id
                or physics["frame_id"] != binding.frame_id
            ):
                mismatch.append("PHYSICS_SAMPLE_BODY_RUN_FRAME_MISMATCH")
            physics_time = physics["time_sec"]
            if type(physics_time) not in (int, float) or not math.isfinite(physics_time):
                raise ValueError("finite independent physics timestamp required")
            nearest = min(samples, key=lambda x: abs(x["sim_time_ns"] / 1e9 - physics_time))
            nearest_models = [x for x in nearest["models"] if x["entity"] == entity]
            # The P0 fixture uses a separately declared identity World/map alignment.
            # A different alignment requires a supported external transform proof.
            if launch["physics_world_frame_alignment"] != "IDENTITY":
                unknown.append("UNSUPPORTED_PHYSICS_WORLD_FRAME_ALIGNMENT")
            if abs(nearest["sim_time_ns"] / 1e9 - physics_time) > 0.2 or len(nearest_models) != 1:
                unknown.append("PHYSICS_ECM_TIME_CORRESPONDENCE_UNVERIFIED")
            else:
                pose = nearest_models[0]["world_pose"]
                observed = _numbers(" ".join(str(physics[k]) for k in ("x", "y", "yaw")), 3)
                if (
                    max(abs(observed[i] - pose[i]) for i in range(2)) > position_tolerance
                    or abs(
                        math.atan2(math.sin(observed[2] - pose[5]), math.cos(observed[2] - pose[5]))
                    )
                    > rotation_tolerance
                ):
                    mismatch.append("PHYSICS_ECM_POSE_MISMATCH")
            reader.unchanged()
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
        ET.ParseError,
        OSError,
    ) as exc:
        unknown.append("UNVERIFIED_SOURCE_FACT:" + str(exc)[:1000])
    evidence["mismatch_reasons"], evidence["unknown_reasons"] = (
        sorted(set(mismatch)),
        sorted(set(unknown)),
    )
    evidence["result"] = "MISMATCH" if mismatch else "UNKNOWN" if unknown else "MATCHED"
    evidence["status"] = {
        "MATCHED": "SIM_SOURCE_MATCHED_NOT_AUTHORIZED",
        "MISMATCH": "SIM_SOURCE_MISMATCH",
        "UNKNOWN": "SIM_SOURCE_UNKNOWN",
    }[evidence["result"]]
    if evidence["result"] == "MATCHED":
        evidence["source_provenance_class"] = "operator_pinned_capture+simulator_observed"
    return evidence
