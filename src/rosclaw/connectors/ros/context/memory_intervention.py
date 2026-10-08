"""Opt-in historical Memory intervention in the actual Native context.

The operator freezes a read-only Core Memory snapshot and reviewed source
admissions. Retrieval and injection are observable, never decision-use or
causal-effect evidence. They grant no capability, permit or current Body fact.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from rosclaw.agentd.context.memory_layer import compile_memory_layer
from rosclaw.agentd.context.sources import EvidenceClass, MemoryItem
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.memory.repository import MemoryRepository

MODES = {"M0", "M1", "M2"}
FACT_FIELDS = {"component", "failure_code", "observed_state", "topic_name", "frame_id"}


class ReadOnlyMemorySnapshot:
    """Minimal read adapter for the real Core repository; no schema migration."""

    def __init__(self, connection):
        self.connection = connection

    def query(self, table, filters=None, limit=100):
        if (
            table not in {"memory_items", "memory_evidence"}
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("bounded Core Memory read query required")
        filters = filters or {}
        if set(filters) - {"id", "memory_id", "body_id", "status"}:
            raise ValueError("explicit Memory source scope required")
        columns = sorted(filters)
        where = " AND ".join('"' + key + '" = ?' for key in columns) or "1"
        rows = self.connection.execute(
            'SELECT * FROM "' + table + '" WHERE ' + where + " LIMIT ?",
            [*[filters[key] for key in columns], limit],
        ).fetchall()
        return [dict(row) for row in rows]


def _owned_path(home, value):
    if type(value) is not str or not 0 < len(value) <= 256 or Path(value).is_absolute():
        raise ValueError("owned relative Memory source path required")
    home = Path(home).resolve()
    path = (home / value).resolve()
    if not path.is_relative_to(home / "data/memory"):
        raise ValueError("Memory experiment source must remain in owned data/memory")
    return path


def _bounded_bytes(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("bounded complete historical source required")
    return raw


def retrieve_intervention(home, body_id, declaration):
    """Read exact admitted Core records; M1 has typed facts, M2 adds patterns."""
    keys = {
        "schema_version",
        "mode",
        "source",
        "approved",
        "database",
        "database_sha256",
        "admissions",
    }
    if (
        type(declaration) is not dict
        or set(declaration) != keys
        or declaration["schema_version"] != "rosclaw.memory_intervention.v1"
    ):
        raise ValueError("closed registered Memory intervention required")
    if (
        type(declaration["mode"]) is not str
        or declaration["mode"] not in MODES
        or declaration["source"] != "operator_reviewed_historical_memory"
        or declaration["approved"] is not True
    ):
        raise ValueError("explicit reviewed historical Memory mode required")
    mode = declaration["mode"]
    result = {
        "schema_version": "rosclaw.memory_intervention_result.v1",
        "status": "RETRIEVED_NOT_DECISION_USE_EVIDENCE",
        "mode": mode,
        "injection_layer": "L5_MEMORY_ADVISORY",
        "authorization": False,
        "current_body_fact": False,
        "physical_acceptance": "NOT_VERIFIED",
        "items": [],
        "referenced_in_decision": False,
        "subsequent_verification": None,
    }
    # Disabled means no database/source reads at all, including absent files.
    if mode == "M0":
        result["status"] = "DISABLED_THIS_MEMORY_SOURCE"
        return result
    path = _owned_path(home, declaration["database"])
    raw = _bounded_bytes(path, 64_000_000)
    if (
        type(declaration["database_sha256"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", declaration["database_sha256"])
        or hashlib.sha256(raw).hexdigest() != declaration["database_sha256"]
        or Path(str(path) + "-wal").exists()
    ):
        raise ValueError("frozen closed read-only Memory snapshot required")
    admissions = declaration["admissions"]
    if type(admissions) is not list or not 1 <= len(admissions) <= 10:
        raise ValueError("one to ten exact reviewed historical sources required")
    seen = set()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        repository = MemoryRepository(ReadOnlyMemorySnapshot(connection))
        for admission in admissions:
            if type(admission) is not dict or set(admission) != {
                "memory_id",
                "source_episode_id",
                "record_sha256",
                "source_path",
                "source_sha256",
                "evidence_class",
                "body_scope",
            }:
                raise ValueError("closed reviewed Memory source admission required")
            if (
                admission["evidence_class"] != "curated"
                or admission["body_scope"] != body_id
                or admission["memory_id"] in seen
            ):
                raise ValueError("reviewed curated source and exact Body scope required")
            seen.add(admission["memory_id"])
            item = repository.get(admission["memory_id"])
            if (
                item is None
                or item.status != "active"
                or item.body_id != body_id
                or item.episode_id != admission["source_episode_id"]
                or digest(item.to_record()) != admission["record_sha256"]
            ):
                raise ValueError("actual Core Memory record/episode/scope differs from admission")
            if not item.episode_id or len(item.episode_id) > 256:
                raise ValueError("original source episode identity required")
            source_path = _owned_path(home, admission["source_path"])
            source_raw = _bounded_bytes(source_path, 2_000_000)
            if hashlib.sha256(source_raw).hexdigest() != admission["source_sha256"]:
                raise ValueError("original historical episode bytes changed")
            original = json.loads(source_raw)
            if type(original) is not dict or original.get("episode_id") != item.episode_id:
                raise ValueError("historical episode source identity differs")
            evidence = repository.evidence_for(item.memory_id)
            if not any(
                e.sha256 == admission["source_sha256"]
                and e.artifact_uri == admission["source_path"]
                and e.source_event_id == item.episode_id
                for e in evidence
            ):
                raise ValueError("Core Memory evidence does not reference admitted episode source")
            data = item.metadata.get("ros_expert_memory")
            if (
                type(data) is not dict
                or set(data) != {"facts", "repair_pattern"}
                or type(data["facts"]) is not dict
                or not data["facts"]
                or set(data["facts"]) - FACT_FIELDS
            ):
                raise ValueError("typed facts separate from historical repair guidance required")
            if any(
                type(v) is not str
                or not 0 < len(v) <= 128
                or not re.fullmatch(r"[A-Za-z0-9_./:-]+", v)
                for v in data["facts"].values()
            ):
                raise ValueError("bounded factual codes only; no free-form instructions in M1")
            entry = {
                "memory_ref": item.memory_id,
                "source_episode_id": item.episode_id,
                "evidence_class": "curated",
                "body_scope": item.body_id,
                "source_sha256": admission["source_sha256"],
                "facts": data["facts"],
                "retrieval_id": "retrieval_"
                + digest(
                    {
                        "mode": mode,
                        "record": admission["record_sha256"],
                        "source": admission["source_sha256"],
                    }
                ),
            }
            if mode == "M2":
                pattern = data["repair_pattern"]
                if type(pattern) is not str or not 0 < len(pattern) <= 2048:
                    raise ValueError("bounded reviewed historical repair pattern required")
                entry["repair_pattern"] = pattern
            result["items"].append(entry)
    if path.read_bytes() != raw:
        raise ValueError("frozen Memory snapshot changed during retrieval")
    result["retrieval_hash"] = digest(result)
    entries = [
        MemoryItem(
            ref=row["memory_ref"],
            summary=json.dumps(
                {
                    "facts": row["facts"],
                    **(
                        {"repair_pattern": row["repair_pattern"]} if "repair_pattern" in row else {}
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            evidence_class=EvidenceClass.CURATED,
            body_scope=row["body_scope"],
        )
        for row in result["items"]
    ]
    _, result["layer_summary"], excluded = compile_memory_layer(entries, body_id=body_id)
    if excluded:
        raise ValueError("historical Memory scope differs from compiled L5 layer")
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > 8000:
        raise ValueError("complete Memory intervention exceeds the Native tool context budget")
    return result


def native_memory_intervention(service, mission):
    """Opt-in actual Native envelope hook; failed sources stay UNKNOWN."""
    configuration = (
        getattr(getattr(service, "_config", None), "raw", {}).get("agent", {}).get("ros_expert", {})
    )
    declaration = configuration.get("memory_intervention")
    if declaration is None:
        return None
    try:
        if declaration.get("schema_version") == "rosclaw.memory_worker_projection.v1":
            from rosclaw.connectors.ros.context.memory_worker_projection import (
                retrieve_worker_projection,
            )

            return retrieve_worker_projection(
                service._home, mission.body_binding.body_id, declaration
            )
        return retrieve_intervention(service._home, mission.body_binding.body_id, declaration)
    except Exception as exc:
        return {
            "status": "UNKNOWN",
            "injection_layer": "L5_MEMORY_ADVISORY",
            "items": [],
            "authorization": False,
            "error": str(exc)[:300],
            "referenced_in_decision": False,
            "subsequent_verification": None,
        }


def retain_injection_evidence(home, envelope):
    """Public per-context injection evidence, with no private prompt/reasoning."""
    from rosclaw.contracts.agent.memory_use import MemoryUseEvidenceV1

    summary = envelope.memory_summary
    if summary.get("injection_layer") != "L5_MEMORY_ADVISORY" or "mode" not in summary:
        return
    records = [
        MemoryUseEvidenceV1(
            retrieval_id=row["retrieval_id"],
            memory_ref=row["memory_ref"],
            source_episode_id=row["source_episode_id"],
            source_sha256=row["source_sha256"],
            evidence_class=row["evidence_class"],
            context_bundle_hash=envelope.hash,
        ).model_dump(mode="json")
        for row in summary["items"]
    ]
    raw = (
        json.dumps(
            {
                "mission_id": envelope.mission_id,
                "context_bundle_hash": envelope.hash,
                "generated_at": envelope.generated_at,
                "mode": summary["mode"],
                "evidence_role": "INJECTION_ONLY_NOT_ADOPTION_OR_CAUSAL_BENEFIT",
                "records": records,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    if len(raw) > 65536:
        raise ValueError("bounded public Memory injection evidence required")
    root = Path(home) / "agentd/memory-contexts"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (digest(raw.decode()) + ".json")
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if _bounded_bytes(path, 65536) != raw:
            raise ValueError("retained Memory injection evidence changed") from None


def tool_memory_correspondence(service, request, result):
    """Exact references in actual tool arguments; never infer benefit from prose."""
    from datetime import UTC, datetime

    from rosclaw.contracts.agent.memory_use import MemoryUseEvidenceV1

    if request.tool_name == "rosclaw_memory_query" or not result.ok:
        return []
    context = getattr(service, "_ros_memory_envelopes", {}).get(request.mission_id)
    if not context:
        return []
    generated = datetime.fromisoformat(context["generated_at"])
    expires = datetime.fromisoformat(context["expires_at"])
    requested = datetime.fromisoformat(request.requested_at)
    mission = service.get_mission(request.mission_id)
    if (
        any(t.tzinfo is None for t in (generated, expires, requested))
        or not generated <= requested < expires
        or expires <= datetime.now(UTC)
        or context["mission_id"] != request.mission_id
        or context["context_revision"] != request.context_revision
        or mission is None
        or context["body"]["body_id"] != mission.body_binding.body_id
        or context["body"]["effective_body_hash"] != mission.body_binding.effective_body_hash
    ):
        return []
    from rosclaw.agentd.pi_bridge.context import envelope_hash
    from rosclaw.contracts.pi.embodied_context import EmbodiedContextEnvelopeV1

    if envelope_hash(EmbodiedContextEnvelopeV1.model_validate(context)) != context["hash"]:
        raise ValueError("actual retained Native envelope content hash changed")
    summary = context.get("memory_summary", {})
    if summary.get("status") != "RETRIEVED_NOT_DECISION_USE_EVIDENCE":
        return []
    arguments = json.dumps(request.arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if len(arguments.encode()) > 65536:
        raise ValueError("bounded actual decision arguments required")
    records = []
    for row in summary.get("items", []):
        if not re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(row["memory_ref"]) + r"(?![A-Za-z0-9_])", arguments
        ):
            continue
        payload = {
            "request_id": request.request_id,
            "mission_id": request.mission_id,
            "pi_session_id": request.pi_session_id,
            "tool_name": request.tool_name,
            "arguments_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
            "ok": result.ok,
            "status": result.status,
            "summary_sha256": hashlib.sha256(result.summary.encode()).hexdigest(),
            "context_bundle_hash": context["hash"],
            "memory_ref": row["memory_ref"],
        }
        record = MemoryUseEvidenceV1(
            retrieval_id=row["retrieval_id"],
            memory_ref=row["memory_ref"],
            source_episode_id=row["source_episode_id"],
            source_sha256=row["source_sha256"],
            evidence_class=row["evidence_class"],
            context_bundle_hash=context["hash"],
            referenced_in_decision=True,
            decision_request_id=request.request_id,
            decision_tool=request.tool_name,
            actual_tool_event_hash=digest(payload),
        )
        records.append({"evidence": record.model_dump(mode="json"), "tool_correspondence": payload})
    return records
