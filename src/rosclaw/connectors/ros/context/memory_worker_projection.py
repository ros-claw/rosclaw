"""Project reviewed historical Memory before an isolated Native worker starts.

Raw historical databases and episode files remain in the operator workspace.
This closes a file-copy leak, not all tool/OS isolation or a causal experiment.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import yaml

from rosclaw.agentd.context.memory_layer import compile_memory_layer
from rosclaw.agentd.context.sources import EvidenceClass, MemoryItem
from rosclaw.connectors.ros.context.memory_intervention import (
    FACT_FIELDS,
    MODES,
    _bounded_bytes,
    _owned_path,
    retrieve_intervention,
)
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def worker_runtime_overrides(declaration):
    """Disable other historical retrieval channels equally in all three arms."""
    if (
        type(declaration) is not dict
        or set(declaration) != {"schema_version", "mode", "retrieval_source"}
        or declaration["schema_version"] != "rosclaw.memory_worker_runtime.v1"
        or type(declaration["mode"]) is not str
        or declaration["mode"] not in MODES
        or declaration["retrieval_source"] != "worker_projection_only"
    ):
        raise ValueError("closed worker-only Memory runtime intervention required")
    from rosclaw.config.models import KnowledgeConfig

    return {
        "enable_memory": False,
        "enable_knowledge": False,
        "enable_how": False,
        "enable_recovery_loop": False,
        "knowledge_v2_mode": "disabled",
        "knowledge": KnowledgeConfig(enabled=False, mode="disabled", url=None, api_key=None),
    }


def _validate_projection(result, mode, body_id):
    fields = {
        "schema_version",
        "status",
        "mode",
        "injection_layer",
        "authorization",
        "current_body_fact",
        "physical_acceptance",
        "items",
        "referenced_in_decision",
        "subsequent_verification",
        "retrieval_hash",
        "layer_summary",
    }
    if type(result) is not dict or set(result) != fields:
        raise ValueError("closed projected Memory result required")
    if (
        result["schema_version"] != "rosclaw.memory_intervention_result.v1"
        or result["mode"] != mode
        or mode not in MODES
        or result["injection_layer"] != "L5_MEMORY_ADVISORY"
        or result["authorization"] is not False
        or result["current_body_fact"] is not False
        or result["referenced_in_decision"] is not False
        or result["subsequent_verification"] is not None
        or result["physical_acceptance"] != "NOT_VERIFIED"
        or type(result["items"]) is not list
        or len(result["items"]) > 10
    ):
        raise ValueError("unexecuted advisory Memory result required")
    if mode == "M0":
        if (
            result["items"]
            or result["layer_summary"]
            or result["status"] != "DISABLED_THIS_MEMORY_SOURCE"
        ):
            raise ValueError("M0 projection must be empty")
    elif result["status"] != "RETRIEVED_NOT_DECISION_USE_EVIDENCE" or not result["items"]:
        raise ValueError("admitted projected historical items required")
    texts = []
    for row in result["items"]:
        required = {
            "memory_ref",
            "source_episode_id",
            "evidence_class",
            "body_scope",
            "source_sha256",
            "facts",
            "retrieval_id",
        }
        if mode == "M2":
            required.add("repair_pattern")
        if type(row) is not dict or set(row) != required:
            raise ValueError("projected facts and guidance must remain separate")
        if row["body_scope"] != body_id or row["evidence_class"] != "curated":
            raise ValueError("exact projected Body scope and evidence class required")
        for field in ("memory_ref", "source_episode_id", "retrieval_id"):
            if type(row[field]) is not str or not 0 < len(row[field]) <= 256:
                raise ValueError("bounded projected source identity required")
        if type(row["source_sha256"]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", row["source_sha256"]
        ):
            raise ValueError("original reviewed source digest required")
        facts = row["facts"]
        if (
            type(facts) is not dict
            or not facts
            or set(facts) - FACT_FIELDS
            or any(
                type(value) is not str
                or not 0 < len(value) <= 128
                or not re.fullmatch(r"[A-Za-z0-9_./:-]+", value)
                for value in facts.values()
            )
        ):
            raise ValueError("bounded factual codes required")
        if mode == "M2" and (
            type(row["repair_pattern"]) is not str or not 0 < len(row["repair_pattern"]) <= 2048
        ):
            raise ValueError("bounded reviewed guidance required")
        texts.append(
            MemoryItem(
                ref=row["memory_ref"],
                summary=json.dumps(
                    {
                        "facts": facts,
                        **({"repair_pattern": row["repair_pattern"]} if mode == "M2" else {}),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                evidence_class=EvidenceClass.CURATED,
                body_scope=body_id,
            )
        )
    _, expected, excluded = compile_memory_layer(texts, body_id=body_id)
    if excluded or result["layer_summary"] != (expected if mode != "M0" else ""):
        raise ValueError("projected advisory text differs from its admitted facts")


def prepare_worker_projection(operator_home, body_id, declaration, destination):
    """Create only filtered JSON, not a raw database or source-episode copy."""
    result = retrieve_intervention(operator_home, body_id, declaration)
    if result["mode"] == "M0":
        result.update(layer_summary="", retrieval_hash=digest(result))
    _validate_projection(result, declaration["mode"], body_id)
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(raw) > 8000:
        raise ValueError("bounded complete worker projection required")
    destination = Path(destination)
    destination.mkdir(exist_ok=False, mode=0o700)
    path = destination / "projection.json"
    path.write_bytes(raw)
    path.chmod(0o400)
    return {
        "schema_version": "rosclaw.memory_worker_projection.v1",
        "source": "operator_projected_historical_memory",
        "approved": True,
        "mode": declaration["mode"],
        "body_id": body_id,
        "projection_path": "data/memory/projection.json",
        "projection_sha256": hashlib.sha256(raw).hexdigest(),
    }


def retrieve_worker_projection(home, body_id, declaration):
    keys = {
        "schema_version",
        "source",
        "approved",
        "mode",
        "body_id",
        "projection_path",
        "projection_sha256",
    }
    if (
        type(declaration) is not dict
        or set(declaration) != keys
        or declaration["schema_version"] != "rosclaw.memory_worker_projection.v1"
        or declaration["source"] != "operator_projected_historical_memory"
        or declaration["approved"] is not True
        or declaration["mode"] not in MODES
        or declaration["body_id"] != body_id
    ):
        raise ValueError("exact operator-projected worker Memory declaration required")
    path = _owned_path(home, declaration["projection_path"])
    raw = _bounded_bytes(path, 8000)
    if hashlib.sha256(raw).hexdigest() != declaration["projection_sha256"]:
        raise ValueError("worker projection differs from frozen operator source")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate worker projection key")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=unique)
    _validate_projection(result, declaration["mode"], body_id)
    if path.read_bytes() != raw:
        raise ValueError("worker projection changed during retrieval")
    return deepcopy(result)


def prepare_registered_worker_sources(operator_home, body_id, declaration, worker_home):
    """Materialize matched Native/MCP source declarations in an exclusive home.

    Caller must independently isolate and mount this home, route every MCP
    child to its runtime profile, and verify the manifest before launch.
    """
    worker_home = Path(worker_home)
    worker_home.mkdir(exist_ok=False, mode=0o700)
    (worker_home / "data").mkdir(mode=0o700)
    projection = prepare_worker_projection(
        operator_home, body_id, declaration, worker_home / "data/memory"
    )
    runtime = {
        "schema_version": "rosclaw.memory_worker_runtime.v1",
        "mode": projection["mode"],
        "retrieval_source": "worker_projection_only",
    }
    worker_runtime_overrides(runtime)
    profile = worker_home / "runtime.yaml"
    profile.write_text(yaml.safe_dump({"ros_expert_memory_experiment": runtime}))
    profile.chmod(0o400)
    native_patch = {"agent": {"ros_expert": {"memory_intervention": projection}}}
    (worker_home / "native-memory-config-patch.json").write_text(
        json.dumps(native_patch, indent=2) + "\n"
    )
    manifest = {
        "schema_version": "rosclaw.memory_worker_sources.v1",
        "body_id": body_id,
        "mode": projection["mode"],
        "native_agent_patch": native_patch,
        "MCP_project_root": str(worker_home.resolve()),
        "MCP_runtime_profile": "runtime.yaml",
        "files": {
            str(p.relative_to(worker_home)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(worker_home.rglob("*"))
            if p.is_file()
        },
        "requires_independent_container_mount_and_tool_route_admission": True,
        "complete_OS_tool_isolation_verified": False,
        "causal_memory_benefit_verified": False,
        "authorization": False,
    }
    manifest["artifact_hash"] = digest(manifest)
    (worker_home / "worker-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
