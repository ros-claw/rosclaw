"""Offline source-byte/declared-consumer contracts, never live physics admission."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from xml.dom import Node, minidom

from pydantic import Field, StrictStr

from rosclaw.connectors.ros.context.source_friction import (
    parse_source_xml,
    source_friction_fact,
)
from rosclaw.connectors.ros.intelligence.graph_contracts import (
    GraphRecord,
    ProvenanceEdgeV1,
    RosConstraintFindingV1,
    RosOperationalGraphV1,
)
from rosclaw.contracts.common import content_hash

INVARIANT = "source.declared_literal_consumer_contract"
SCOPE = "STATIC_DECLARED_PARSER_CONTRACT_ONLY"


class RosLiteralConsumerContractV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_literal_consumer_contract.v1"] = (
        "rosclaw.ros_literal_consumer_contract.v1"
    )
    contract_hash: StrictStr
    sdk_image: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    library_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    slot: Literal["SDF_FRICTION_DIRECTION_FRAME"] = "SDF_FRICTION_DIRECTION_FRAME"
    literal_attribute: Literal["gz:expressed_in"] = "gz:expressed_in"
    run_witness_plugin: StrictStr = Field(
        default="rosclaw::SourceWitness", min_length=1, max_length=256
    )
    parser_evidence_ref: StrictStr = Field(min_length=1, max_length=2048)
    parser_evidence_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    assurance: Literal["DECLARED_OPERATOR_CONTRACT_NOT_RUNTIME_AUTHENTICATION"] = (
        "DECLARED_OPERATOR_CONTRACT_NOT_RUNTIME_AUTHENTICATION"
    )

    @classmethod
    def from_dict(cls, value):
        contract = cls.model_validate(value)
        payload = contract.model_dump(mode="json")
        digest = payload.pop("contract_hash")
        if digest != content_hash("rosconsumercontract", payload):
            raise ValueError("source consumer contract hash mismatch")
        return contract


def _json(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 5_000_000:
        raise ValueError("bounded original JSON bytes required")
    return json.loads(raw)


def _children(nodes, tag):
    return [
        child
        for node in nodes
        for child in node.childNodes
        if child.nodeType == Node.ELEMENT_NODE and child.tagName == tag
    ]


def _friction_spellings(urdf, world, world_bytes, entity):
    names = {link.get("name") for link in urdf.findall("link")}
    source = []
    for block in urdf.findall("gazebo"):
        for ode in block.findall("collision/surface/friction/ode"):
            fact = source_friction_fact(block.get("reference"), ode, names)
            if fact is not None:
                if fact[0] not in names:
                    raise ValueError("original friction references an unknown link")
                source.append(fact)
    models = [m for m in world.findall("world/model") if m.get("name") == entity]
    if len(models) != 1:
        raise ValueError("one declared source model required")
    loaded = {link.get("name") for link in models[0].findall("link")}
    if any(fact[2] not in loaded for fact in source):
        raise ValueError("source frame missing from loaded SDF links")
    actual = []
    for link in models[0].findall("link"):
        for ode in link.findall("collision/surface/friction/ode"):
            fact = source_friction_fact(link.get("name"), ode, names)
            if fact is not None:
                actual.append(fact)
    if not source or len(source) > 128 or sorted(source) != sorted(actual):
        raise ValueError("derived friction facts differ from original source or are absent")
    with minidom.parseString(world_bytes) as document:
        models = [
            m
            for m in document.getElementsByTagName("model")
            if m.parentNode.nodeName == "world" and m.getAttribute("name") == entity
        ]
        nodes = models
        for tag in ("link", "collision", "surface", "friction", "ode", "fdir1"):
            nodes = _children(nodes, tag)
        spellings = []
        for node in nodes:
            attrs = [
                attribute
                for attribute in node.attributes.values()
                if attribute.localName == "expressed_in"
            ]
            if len(attrs) != 1:
                raise ValueError("one original friction frame attribute required")
            spellings.append(attrs[0].name)
    if len(spellings) != len(actual):
        raise ValueError("ambiguous literal friction declaration coverage")
    return spellings, source


def inspect_source_consumer(
    graph,
    urdf_bytes,
    world_bytes,
    contract_value,
    container_bytes,
    backend_bytes,
    parser_evidence_bytes,
    *,
    robot_entity,
):
    """Compute a new diagnostic graph from originals; caller cannot supply a verdict.

    A contract is operator-declared knowledge, not authentication of an SDK or
    proof that a running backend consumed a field. Version mismatches stay
    UNKNOWN. No input is adapted and no file, ROS node or container is opened.
    """
    graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
    contract = RosLiteralConsumerContractV1.from_dict(contract_value)
    _json(parser_evidence_bytes)
    if hashlib.sha256(parser_evidence_bytes).hexdigest() != contract.parser_evidence_sha256:
        raise ValueError("original parser evidence differs from the declared contract")
    if graph.evidence_domain != "SEALED_REPLAY" or graph.source_status == "MISMATCH":
        raise ValueError("original sealed replay graph without source contradiction required")
    if type(robot_entity) is not str or not 0 < len(robot_entity) <= 256:
        raise ValueError("bounded explicit source entity required")
    for raw in (urdf_bytes, world_bytes):
        if type(raw) is not bytes or not 0 < len(raw) <= 4_000_000:
            raise ValueError("bounded original XML bytes required")
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise ValueError("DTD and entity declarations are outside source contract scope")
    try:
        urdf, world = parse_source_xml(urdf_bytes), parse_source_xml(world_bytes)
    except SyntaxError as exc:
        raise ValueError("invalid original source XML") from exc
    if urdf.tag != "robot" or world.tag != "sdf":
        raise ValueError("original URDF and world SDF required")
    urdf_hash = hashlib.sha256(urdf_bytes).hexdigest()
    if graph.source_urdf_sha256 != urdf_hash:
        raise ValueError("source URDF differs from the original graph")
    runs = {e.name for e in graph.entities if e.kind == "run"}
    witnesses = [
        plugin
        for plugin in world.findall("world/plugin")
        if plugin.get("name") == contract.run_witness_plugin
    ]
    world_runs = {plugin.findtext("run_id") for plugin in witnesses}
    if runs and (len(runs) != 1 or len(witnesses) != 1 or runs != world_runs):
        raise ValueError("SDF run identity differs from the original graph")
    container, backend = _json(container_bytes), _json(backend_bytes)
    if isinstance(container, list) and len(container) == 1:
        container = container[0]
    if type(container) is not dict or type(backend) is not dict:
        raise ValueError("one original container inspection and backend record required")
    image = container.get("Image")
    library = backend.get("library_sha256")
    status, reasons, spellings, facts = "UNKNOWN", [], [], []
    if image != contract.sdk_image or library != contract.library_sha256:
        reasons.append("SDK_OR_LIBRARY_DIFFERS_FROM_DECLARED_CONSUMER_CONTRACT")
    else:
        try:
            spellings, facts = _friction_spellings(urdf, world, world_bytes, robot_entity)
        except ValueError as exc:
            reasons.append("SOURCE_CONSUMPTION_PRECONDITION_UNRESOLVED: " + str(exc)[:240])
        else:
            status = (
                "SATISFIED"
                if all(name == contract.literal_attribute for name in spellings)
                else "VIOLATED"
            )
            reasons.append(
                "ALL_DECLARED_SOURCE_ATTRIBUTES_HAVE_CONSUMER_LITERAL_SPELLING"
                if status == "SATISFIED"
                else "EXPANDED_SOURCE_FRAME_PRESERVED_BUT_LITERAL_CONSUMER_ATTRIBUTE_MISSING"
            )
    inputs = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in [
            ("source_urdf", urdf_bytes),
            ("world_sdf", world_bytes),
            ("container_inspection", container_bytes),
            ("backend_observation", backend_bytes),
            ("parser_evidence", parser_evidence_bytes),
        ]
    }
    review = {
        "schema_version": "rosclaw.ros_source_consumer_review.v1",
        "original_graph_hash": graph.graph_hash,
        "source_ref": graph.source_ref,
        "snapshot_hash": graph.snapshot_hash,
        "source_urdf_sha256": urdf_hash,
        "reviewed_at": graph.reviewed_at.isoformat(),
        "evidence_domain": "SEALED_REPLAY",
        "contract": contract.model_dump(mode="json"),
        "input_sha256": inputs,
        "robot_entity": robot_entity,
        "run_binding": "ORIGINAL_GRAPH_RUN_MATCHED" if runs else "ABSENT_NO_CURRENT_IDENTITY",
        "status": status,
        "reasons": reasons,
        "attribute_spellings": spellings,
        "original_friction_facts": facts,
        "scope": SCOPE,
        "current_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
        "runtime_consumption": "NOT_ATTESTED_BY_OFFLINE_BYTES",
        "causal_verification": "NOT_ESTABLISHED",
        "physical_stop": "NOT_EVALUATED",
    }
    review["review_hash"] = content_hash("rosconsumerreview", review)
    payload = graph.model_dump(mode="json")
    payload.pop("graph_hash")
    if any(f.invariant == INVARIANT for f in graph.findings):
        raise ValueError("consumer finding already attached; retain the original base graph")
    evidence = {
        "document_ref": review["review_hash"],
        "document_hash": review["review_hash"],
        "pointer": "/status",
        "value_hash": content_hash("opvalue", status),
        "observed_at": graph.reviewed_at.isoformat(),
        # Static source evidence does not become a fresh dynamic measurement.
        "freshness": "UNKNOWN",
    }
    ref = ProvenanceEdgeV1(id=content_hash("opref", evidence), **evidence)
    finding = RosConstraintFindingV1(
        id=content_hash("rosconsumerfinding", [contract.contract_hash, robot_entity]),
        invariant=INVARIANT,
        scope=SCOPE,
        status=status,
        targets=[robot_entity],
        evidence_refs=[ref.id],
        missing_evidence=reasons if status == "UNKNOWN" else [],
    )
    payload["provenance"][ref.id] = ref.model_dump(mode="json")
    payload["findings"].append(finding.model_dump(mode="json"))
    result = RosOperationalGraphV1.from_dict(
        {**payload, "graph_hash": content_hash("rosopgraph", payload)}
    )
    return review, result
