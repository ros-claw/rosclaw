"""Equal original evidence access for every future ACI arm; no ROS I/O.

The view projects measured fields, never fault IDs, constraint verdicts, repair
candidates or gold answers. Original sources are sealed and content-addressed.
Oversized records remain explicitly expandable instead of silently disappearing.
"""

import copy
import hashlib
import json
from typing import Literal

import yaml

from rosclaw.contracts.common import content_hash

EvidenceView = Literal[
    "task_requirements",
    "interfaces",
    "task_interfaces",
    "navigation_parameters",
    "all_parameters",
    "signals",
    "transforms",
    "lifecycle",
    "staging",
    "body_geometry",
    "robot_urdf",
]
NAVIGATION_PARAMETERS = frozenset(
    {"tf_broadcast", "node_names", "robot_radius", "footprint", "robot_base_frame"}
)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pointer(*parts):
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


class CommonEvidence:
    def __init__(self, aci):
        self.aci = aci
        self.groups = {name: [] for name in EvidenceView.__args__}
        self.records = {}
        self.model = aci.model.to_dict()
        self.model_ref = "sealed-model:" + aci.model.snapshot_hash
        for name in ("task_requirements",):
            self.add(
                name, aci.requirements, "original-task:requirements.json", "", "MEASURED_TASK_INPUT"
            )
        for kind in ("topics", "actions"):
            for index, value in enumerate(self.model["graph"].get(kind, [])):
                entry = self.add("interfaces", value, self.model_ref, pointer("graph", kind, index))
                if value.get("name") in aci.requirements.get("interface_names", {}).values():
                    self.groups["task_interfaces"].append(entry)
        for node, values in sorted(self.model["observations"].get("node_parameters", {}).items()):
            for name, value in sorted(values.items()):
                entry = self.add(
                    "all_parameters",
                    value,
                    self.model_ref,
                    pointer("observations", "node_parameters", node, name),
                    node=node,
                    parameter=name,
                    observed_at=self.model["observations"]
                    .get("parameter_captured_at", {})
                    .get(node),
                )
                if name in NAVIGATION_PARAMETERS:
                    self.groups["navigation_parameters"].append(entry)
        for group in ("signals", "transforms", "lifecycle"):
            for index, value in enumerate(self.model.get(group, [])):
                self.add(group, value, self.model_ref, pointer(group, index))
        for name in ("nav2.yaml", "bringup.json"):
            self.add(
                "staging",
                yaml.safe_load(aci.tools.files[name]),
                "original-task:" + name,
                "",
                "MEASURED_TASK_INPUT",
                file=name,
            )
        geometry = aci.case.derived_documents["navigation_observation_review"]["candidate"][
            "geometry"
        ]
        # Identical offline Body geometry in all arms, not an Expert repair verdict.
        self.add(
            "body_geometry",
            geometry,
            content_hash("derived", geometry),
            "",
            "INFERRED_OFFLINE_URDF",
        )
        self.add(
            "robot_urdf",
            aci.tools.files["robot.urdf"],
            "original-task:robot.urdf",
            "",
            "MEASURED_TASK_INPUT",
        )
        self.common_hash = content_hash("acicommon", self.groups)

    def add(
        self,
        group,
        value,
        source_ref,
        source_pointer,
        classification="MEASURED_SNAPSHOT",
        **metadata,
    ):
        payload = {
            "value": copy.deepcopy(value),
            "source_ref": source_ref,
            "pointer": source_pointer,
            "value_hash": content_hash("factvalue", value),
            "classification": classification,
            **metadata,
        }
        ref = content_hash("acievidence", payload)
        entry = {"evidence_ref": ref, **payload}
        self.records[ref] = entry
        self.groups[group].append(entry)
        return entry

    def bound(self):
        self.aci._bound(self.aci.case.case_hash)
        if content_hash("acicommon", self.groups) != self.common_hash:
            raise ValueError("common evidence changed after compilation")
        if self.aci.model.snapshot_hash != self.aci.model.compute_snapshot_hash():
            raise ValueError("original observation snapshot changed")

    def catalog(self):
        self.bound()
        return {
            "common_observation_hash": self.common_hash,
            "captured_at": self.aci.model.captured_at.isoformat(),
            "views": {name: len(rows) for name, rows in self.groups.items()},
            "scope": "SEALED_ORIGINAL_NOT_CURRENT_L1",
            "current_authority": "NONE",
        }

    def read(self, view, cursor):
        self.bound()
        rows = self.groups[view]
        if not 0 <= cursor <= len(rows):
            raise ValueError("evidence cursor outside declared view")
        page = []
        index = cursor
        while index < len(rows) and len(page) < 16:
            entry = rows[index]
            if len(encoded(entry)) > 1600:
                entry = {
                    key: entry[key]
                    for key in (
                        "evidence_ref",
                        "source_ref",
                        "pointer",
                        "value_hash",
                        "classification",
                    )
                }
                entry.update(
                    {
                        "value_omitted": True,
                        "serialized_characters": len(encoded(rows[index])),
                        "expand_operation": "ros_engineer.expand_task_evidence",
                    }
                )
            if page and len(encoded(page + [entry])) > 2000:
                break
            page.append(copy.deepcopy(entry))
            index += 1
        return {
            "status": "SEALED_ORIGINAL_EVIDENCE_NOT_ADMISSION",
            "view": view,
            "records": page,
            "total_records": len(rows),
            "observation_state": "PRESENT" if rows else "UNKNOWN_NO_RECORDS",
            "next_cursor": index if index < len(rows) else None,
            "snapshot_hash": self.aci.case.snapshot_hash,
            "common_observation_hash": self.common_hash,
            "completeness": copy.deepcopy(self.aci.model.completeness),
            "current_authority": "NONE",
        }

    def expand(self, ref, offset, limit):
        self.bound()
        if ref not in self.records:
            raise ValueError("select an evidence_ref actually returned by a common view")
        raw = encoded(self.records[ref])
        if offset > len(raw):
            raise ValueError("evidence offset outside original serialized record")
        return {
            "status": "ORIGINAL_SERIALIZED_RECORD_SLICE_NOT_NEW_OBSERVATION",
            "evidence_ref": ref,
            "record_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "offset": offset,
            "total_characters": len(raw),
            "text": raw[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(raw) else None,
            "current_authority": "NONE",
        }
