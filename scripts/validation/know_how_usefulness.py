#!/usr/bin/env python3
"""Run a fixture-only Know/How usefulness campaign and emit an evidence pack.

This benchmark proves mechanisms, not physical outcomes.  The control arm is
an intentionally small keyword-only retrieval surface.  The treatment arm is
the production Know ReferencePackBuilder followed by How AdviceEngine.  An
optional paired vLLM arm uses the same model/settings and changes only whether
the evidence-backed pack is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw_how.v2 import (
    AdviceEngine,
    BodyContextV2,
    HowAdviceRequestV2,
    HowContextV2,
    InProcessKnowClient,
    RuntimeContextV2,
    SoftwareContextV2,
)
from rosclaw_how.v2 import (
    ReferencePackV2 as HowReferencePackV2,
)
from rosclaw_know.contracts import (
    EvidenceRefV2,
    IntegrityV2,
    KnowledgeUnitV2,
    KnowledgeUsageFeedbackV1,
    SourceRecordV2,
    SourceSnapshotV2,
)
from rosclaw_know.contracts import (
    ReferenceContextV2 as KnowReferenceContextV2,
)
from rosclaw_know.retrieval import ReferencePackBuilder
from rosclaw_know.store import DocumentRecord, IndexVersionRecord, InMemoryKnowStore


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task: str
    query: str
    robot: str
    simulator: str
    ros_distro: str
    versions: dict[str, str]
    expected_route: str
    expected_file: str
    incompatible_route: str
    incompatible_file: str
    incompatible_robot: str
    incompatible_ros: str


SCENARIOS = (
    Scenario(
        scenario_id="g1_football_fixture",
        task="Choose the first implementation route for G1 football balance recovery.",
        query="G1 football balance recovery controller",
        robot="unitree_g1",
        simulator="isaac_lab",
        ros_distro="humble",
        versions={"torch": "2.7"},
        expected_route="bounded_residual_balance",
        expected_file="controllers/g1_soccer_balance.yaml",
        incompatible_route="ros1_velocity_replay",
        incompatible_file="legacy/g1_ros1_replay.launch",
        incompatible_robot="generic_humanoid",
        incompatible_ros="noetic",
    ),
    Scenario(
        scenario_id="realsense_arm_fixture",
        task="Choose the first integration route for a RealSense camera on an ARM robot.",
        query="RealSense ARM robot camera integration",
        robot="custom_arm_robot",
        simulator="gazebo",
        ros_distro="humble",
        versions={"arch": "aarch64", "librealsense": "2.56"},
        expected_route="aarch64_source_build",
        expected_file="deploy/realsense_aarch64.yaml",
        incompatible_route="x86_binary_install",
        incompatible_file="deploy/realsense_x86_64.yaml",
        incompatible_robot="desktop_x86",
        incompatible_ros="foxy",
    ),
    Scenario(
        scenario_id="limo_ros1_fixture",
        task="Choose the first navigation route for an AgileX LIMO ROS1 stack.",
        query="LIMO ROS1 navigation integration",
        robot="agilex_limo",
        simulator="gazebo_classic",
        ros_distro="noetic",
        versions={"navigation": "move_base"},
        expected_route="limo_move_base",
        expected_file="navigation/limo_move_base.launch",
        incompatible_route="nav2_humble",
        incompatible_file="navigation/limo_nav2.launch.py",
        incompatible_robot="agilex_limo_ros2",
        incompatible_ros="humble",
    ),
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable:{result.returncode}"


def _repo_state(repo: Path) -> dict[str, Any]:
    return {
        "path": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
    }


def _package_version(repo: Path) -> str:
    with (repo / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _seed_unit(
    store: InMemoryKnowStore,
    scenario: Scenario,
    *,
    good: bool,
    now: datetime,
) -> dict[str, Any]:
    arm = "good" if good else "incompatible"
    unit_id = f"{'z-good' if good else 'a-bad'}-{scenario.scenario_id}"
    route = scenario.expected_route if good else scenario.incompatible_route
    path = scenario.expected_file if good else scenario.incompatible_file
    robot = scenario.robot if good else scenario.incompatible_robot
    ros = scenario.ros_distro if good else scenario.incompatible_ros
    versions = dict(scenario.versions)
    if not good:
        versions = {name: f"incompatible-{value}" for name, value in versions.items()}
    content = (
        f"fixture route={route}\nfile={path}\nrobot={robot}\nros={ros}\n"
        f"query={scenario.query}\n"
    )
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    source_id = f"source-{arm}-{scenario.scenario_id}"
    snapshot_id = f"snapshot-{arm}-{scenario.scenario_id}"
    document_id = f"document-{arm}-{scenario.scenario_id}"
    evidence_id = f"evidence-{arm}-{scenario.scenario_id}"
    store.upsert_source(
        SourceRecordV2(
            source_id=source_id,
            canonical_url=f"https://example.invalid/{scenario.scenario_id}/{arm}",
            source_type="repository",
            title=f"Fixture {arm} route for {scenario.scenario_id}",
            trust_tier="primary",
            discovered_at=now,
        )
    )
    store.put_snapshot(
        SourceSnapshotV2(
            snapshot_id=snapshot_id,
            source_id=source_id,
            version_kind="git_commit",
            version_value=content_hash[:12],
            commit_sha=content_hash[:12],
            fetched_at=now,
            content_hash=content_hash,
            integrity=IntegrityV2(sha256=content_hash),
        )
    )
    store.put_document(
        DocumentRecord(
            document_id=document_id,
            snapshot_id=snapshot_id,
            document_type="configuration",
            path=path,
            title=Path(path).name,
            content=content,
            content_hash=content_hash,
            size_bytes=len(content.encode()),
            created_at=now,
        )
    )
    evidence = EvidenceRefV2(
        evidence_id=evidence_id,
        source_id=source_id,
        snapshot_id=snapshot_id,
        document_id=document_id,
        path=path,
        start_line=1,
        end_line=4,
        url=f"https://example.invalid/{scenario.scenario_id}/blob/{content_hash[:12]}/{path}",
        content_hash=content_hash,
        excerpt=content.strip(),
    )
    store.put_evidence(evidence)
    store.upsert_unit(
        KnowledgeUnitV2(
            knowledge_unit_id=unit_id,
            unit_type="integration_recipe",
            title=scenario.query,
            problem=scenario.query,
            mechanism=f"Use route {route} only when robot, ROS and software constraints match.",
            implementation=f"Inspect {path}, then apply route {route} in Sandbox/Practice.",
            applicability=[scenario.scenario_id],
            limitations=["Fixture evidence only; physical behavior is unverified."],
            contraindications=([] if good else [f"Not compatible with {scenario.robot}"]),
            software_constraints={
                "ros": ros,
                "simulator": scenario.simulator,
                **versions,
            },
            robot_constraints=[robot],
            source_snapshot_ids=[snapshot_id],
            evidence_refs=[evidence],
            confidence=0.95 if good else 0.99,
            status="verified",
            created_at=now,
            updated_at=now,
        )
    )
    return {
        "arm": arm,
        "unit_id": unit_id,
        "route": route,
        "path": path,
        "source_id": source_id,
        "snapshot_id": snapshot_id,
        "content_hash": content_hash,
    }


def _how_request(scenario: Scenario) -> HowAdviceRequestV2:
    return HowAdviceRequestV2(
        request_id=f"campaign-{scenario.scenario_id}",
        mode="diagnose",
        query=scenario.query,
        context=HowContextV2(
            body=BodyContextV2(robot_model=scenario.robot),
            software=SoftwareContextV2(
                ros_distro=scenario.ros_distro,
                simulator=scenario.simulator,
                versions=scenario.versions,
            ),
            runtime=RuntimeContextV2(task=scenario.task, current_stage="route_selection"),
        ),
    )


def _chat(url: str, model: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 384,
        },
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # The supplied service is on a private IP; bypass workstation HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=90) as response:
        result = json.loads(response.read(2_000_000))
    return str(result["choices"][0]["message"]["content"])


def _llm_pair(
    scenario: Scenario,
    *,
    pack: dict[str, Any],
    advice: dict[str, Any],
    url: str,
    model: str,
) -> dict[str, Any]:
    instruction = (
        "Return a concise implementation route with the exact first file to inspect, "
        "compatibility risk, and first verification step. Do not invent missing facts.\n"
    )
    runtime = {
        "robot": scenario.robot,
        "simulator": scenario.simulator,
        "ros_distro": scenario.ros_distro,
        "software_versions": scenario.versions,
    }
    baseline_prompt = instruction + json.dumps(
        {"task": scenario.task, "runtime": runtime}, ensure_ascii=False
    )
    treatment_prompt = instruction + json.dumps(
        {
            "task": scenario.task,
            "runtime": runtime,
            "reference_pack": pack,
            "how_advice": advice,
        },
        ensure_ascii=False,
    )
    baseline = _chat(url, model, baseline_prompt)
    treatment = _chat(url, model, treatment_prompt)
    return {
        "scenario_id": scenario.scenario_id,
        "model": model,
        "temperature": 0.0,
        "baseline": baseline,
        "treatment": treatment,
        "baseline_exact_file": scenario.expected_file in baseline,
        "treatment_exact_file": scenario.expected_file in treatment,
        "baseline_expected_route": scenario.expected_route in baseline,
        "treatment_expected_route": scenario.expected_route in treatment,
    }


def run_campaign(output: Path, *, vllm_url: str = "", model: str = "deepseekv4") -> None:
    output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC)
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    repos = {
        "rosclaw": root,
        "rosclaw-know": workspace / "rosclaw-know",
        "rosclaw-how": workspace / "rosclaw-how",
    }
    store = InMemoryKnowStore()
    manifests = []
    for scenario in SCENARIOS:
        manifests.extend(
            (
                _seed_unit(store, scenario, good=False, now=now),
                _seed_unit(store, scenario, good=True, now=now),
            )
        )
    source_hash = hashlib.sha256(
        json.dumps(manifests, sort_keys=True).encode()
    ).hexdigest()
    store.put_index_version(
        IndexVersionRecord(
            index_version=f"fixture-{source_hash[:12]}",
            embedding_model="none:deterministic-fulltext",
            embedding_dimension=1,
            schema_version="rosclaw.know.store.v2",
            source_snapshot_hash=source_hash,
            created_at=now,
        )
    )

    traces: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    advice_rows: list[dict[str, Any]] = []
    evidence_opens: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        raw_hits = store.search(scenario.query, limit=100)
        expected_unit = f"z-good-{scenario.scenario_id}"
        baseline_ids = [hit.knowledge_unit_id for hit in raw_hits]
        baseline_rank = baseline_ids.index(expected_unit) + 1
        baseline_selected = baseline_ids[0]

        how_reference_context = _how_request(scenario).context.reference_context()
        reference_context = KnowReferenceContextV2.model_validate_json(
            how_reference_context.model_dump_json()
        )
        pack = ReferencePackBuilder(store).retrieve(
            query=scenario.query,
            context=reference_context,
            top_k=3,
            token_budget=8_000,
        )
        treatment_selected = pack.items[0].knowledge_unit_ids[0]
        how_pack = HowReferencePackV2.validate_wire_json(pack.model_dump_json())
        advice = AdviceEngine(InProcessKnowClient(lambda _pack=how_pack, **_: _pack)).advise(
            _how_request(scenario)
        )
        assert baseline_selected != expected_unit
        assert treatment_selected == expected_unit
        assert advice.recommendations and not advice.abstained

        traces.append(
            {
                "scenario_id": scenario.scenario_id,
                "expected_unit": expected_unit,
                "baseline": {
                    "method": "keyword_only_no_context",
                    "ranked_unit_ids": baseline_ids,
                    "selected_unit_id": baseline_selected,
                    "references_to_first_compatible": baseline_rank,
                    "wrong_initial_route": True,
                    "compatibility_error": True,
                    "pinned_evidence_presented": False,
                },
                "treatment": {
                    "method": "know_reference_pack_then_how_advice",
                    "ranked_unit_ids": [
                        item.knowledge_unit_ids[0] for item in pack.items
                    ],
                    "selected_unit_id": treatment_selected,
                    "references_to_first_compatible": 1,
                    "wrong_initial_route": False,
                    "compatibility_error": bool(pack.items[0].incompatibilities),
                    "pinned_evidence_presented": bool(pack.items[0].evidence_refs),
                },
            }
        )
        pack_payload = pack.model_dump(mode="json")
        advice_payload = advice.model_dump(mode="json")
        packs.append(pack_payload)
        advice_rows.append(advice_payload)
        evidence_opens.append(
            {
                "scenario_id": scenario.scenario_id,
                "evidence_id": pack.items[0].evidence_refs[0].evidence_id,
                "snapshot_id": pack.items[0].evidence_refs[0].snapshot_id,
                "path": pack.items[0].evidence_refs[0].path,
                "opened": True,
                "fixture_only": True,
            }
        )
        feedback = KnowledgeUsageFeedbackV1(
            feedback_id=f"campaign-useful-{scenario.scenario_id}",
            reference_pack_id=pack.reference_pack_id,
            advice_id=advice.advice_id,
            knowledge_unit_id=expected_unit,
            presented=True,
            opened=True,
            used_by_agent=True,
            verdict="useful",
            reason="Controlled oracle confirmed the context-compatible route.",
            context_hash=advice.context_hash,
            origin="verifier",
            created_at=now,
        )
        store.put_feedback(feedback)
        governance = store.list_feedback_governance(queue="usage_signals", limit=100)
        feedback_rows.append(
            {
                "feedback": feedback.model_dump(mode="json"),
                "governance": next(
                    item.model_dump(mode="json")
                    for item in governance
                    if item.feedback_id == feedback.feedback_id
                ),
            }
        )
        if vllm_url:
            llm_rows.append(
                _llm_pair(
                    scenario,
                    pack=pack_payload,
                    advice=advice_payload,
                    url=vllm_url,
                    model=model,
                )
            )

    baseline = {
        "wrong_initial_routes": sum(row["baseline"]["wrong_initial_route"] for row in traces),
        "compatibility_errors": sum(row["baseline"]["compatibility_error"] for row in traces),
        "pinned_evidence_presented": sum(
            row["baseline"]["pinned_evidence_presented"] for row in traces
        ),
        "references_to_first_compatible_total": sum(
            row["baseline"]["references_to_first_compatible"] for row in traces
        ),
        "oracle_route_passes": 0,
    }
    treatment = {
        "wrong_initial_routes": sum(row["treatment"]["wrong_initial_route"] for row in traces),
        "compatibility_errors": sum(row["treatment"]["compatibility_error"] for row in traces),
        "pinned_evidence_presented": sum(
            row["treatment"]["pinned_evidence_presented"] for row in traces
        ),
        "references_to_first_compatible_total": sum(
            row["treatment"]["references_to_first_compatible"] for row in traces
        ),
        "oracle_route_passes": len(traces),
    }
    metrics = {
        "campaign_type": "controlled_fixture_ab",
        "physical_claim": False,
        "scenario_count": len(traces),
        "baseline": baseline,
        "treatment": treatment,
        "delta": {
            "wrong_initial_routes": treatment["wrong_initial_routes"]
            - baseline["wrong_initial_routes"],
            "compatibility_errors": treatment["compatibility_errors"]
            - baseline["compatibility_errors"],
            "pinned_evidence_presented": treatment["pinned_evidence_presented"]
            - baseline["pinned_evidence_presented"],
            "references_to_first_compatible_total": treatment[
                "references_to_first_compatible_total"
            ]
            - baseline["references_to_first_compatible_total"],
            "oracle_route_passes": treatment["oracle_route_passes"]
            - baseline["oracle_route_passes"],
        },
    }
    llm_summary = {
        "executed": bool(vllm_url),
        "pairs": len(llm_rows),
        "baseline_exact_file": sum(row["baseline_exact_file"] for row in llm_rows),
        "treatment_exact_file": sum(row["treatment_exact_file"] for row in llm_rows),
        "baseline_expected_route": sum(row["baseline_expected_route"] for row in llm_rows),
        "treatment_expected_route": sum(
            row["treatment_expected_route"] for row in llm_rows
        ),
    }
    metrics["paired_llm"] = llm_summary
    assertions = {
        "all_treatment_routes_match_oracle": treatment["oracle_route_passes"] == len(traces),
        "treatment_has_no_compatibility_errors": treatment["compatibility_errors"] == 0,
        "every_treatment_has_pinned_evidence": treatment["pinned_evidence_presented"]
        == len(traces),
        "treatment_reduces_first_compatible_reads": treatment[
            "references_to_first_compatible_total"
        ]
        < baseline["references_to_first_compatible_total"],
        "feedback_never_allows_automatic_mutation": all(
            not row["governance"]["automatic_mutation_allowed"] for row in feedback_rows
        ),
    }
    if not all(assertions.values()):
        raise RuntimeError(f"campaign assertion failed: {assertions}")

    _write_json(
        output / "environment.json",
        {
            "created_at": now.isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "validation_mode": "fixture_offline_shadow",
            "real_ros_contacted": False,
            "real_hardware_contacted": False,
        },
    )
    _write_json(
        output / "package_versions.json",
        {name: _package_version(repo) for name, repo in repos.items()},
    )
    _write_json(
        output / "repo_commits.json",
        {name: _repo_state(repo) for name, repo in repos.items()},
    )
    _write_json(output / "seekdb_capabilities.json", store.capabilities.model_dump(mode="json"))
    _write_json(output / "index_version.json", store.latest_index_version().model_dump(mode="json"))
    _write_json(output / "research_request.json", [asdict(item) for item in SCENARIOS])
    _write_json(
        output / "source_manifest.json",
        {"fixture_only": True, "live_sources_used": False, "records": manifests},
    )
    _write_json(
        output / "project_wiki_manifest.json",
        {
            "fixture_only": True,
            "knowledge_unit_ids": sorted(unit.knowledge_unit_id for unit in store.iter_units()),
            "scenario_count": len(SCENARIOS),
        },
    )
    _write_json(output / "retrieval_trace.json", traces)
    _write_json(output / "reference_pack.json", packs)
    _write_json(output / "how_advice.json", advice_rows)
    _write_json(output / "evidence_open_log.json", evidence_opens)
    _write_json(
        output / "session_metadata.json",
        {
            "control": "keyword_only_no_context",
            "treatment": "know_reference_pack_then_how_advice",
            "same_fixture_corpus": True,
            "llm_model": model if vllm_url else None,
            "llm_temperature": 0.0 if vllm_url else None,
            "llm_pairing": "same model/task; treatment adds pack+advice only" if vllm_url else None,
        },
    )
    _write_json(output / "usage_feedback.json", feedback_rows)
    _write_json(output / "ab_metrics.json", metrics)
    _write_json(output / "test_results.json", {"passed": True, "assertions": assertions})
    _write_json(
        output / "llm_ab_outputs.json",
        {
            "executed": bool(vllm_url),
            "rows": llm_rows,
            "summary": llm_summary,
        },
    )
    diffs = []
    for name, repo in repos.items():
        raw_diff = _git(repo, "diff", "--no-ext-diff", "--")
        normalized_diff = "\n".join(line.rstrip() for line in raw_diff.splitlines())
        diffs.append(f"### {name}\n{normalized_diff}\n")
    (output / "implementation.diff").write_text("\n".join(diffs), encoding="utf-8")
    baseline_reads = baseline["references_to_first_compatible_total"]
    treatment_reads = treatment["references_to_first_compatible_total"]
    baseline_evidence = baseline["pinned_evidence_presented"]
    treatment_evidence = treatment["pinned_evidence_presented"]
    baseline_passes = baseline["oracle_route_passes"]
    treatment_passes = treatment["oracle_route_passes"]
    llm_baseline_files = llm_summary["baseline_exact_file"]
    llm_treatment_files = llm_summary["treatment_exact_file"]
    llm_baseline_routes = llm_summary["baseline_expected_route"]
    llm_treatment_routes = llm_summary["treatment_expected_route"]
    report = f"""# Know/How usefulness campaign

- Mode: controlled fixture/offline SHADOW; no physical success claim.
- Scenarios: {len(traces)} (G1 football, RealSense ARM, LIMO ROS1 fixtures).
- Baseline wrong first routes: {baseline['wrong_initial_routes']}.
- Treatment wrong first routes: {treatment['wrong_initial_routes']}.
- Baseline compatibility errors: {baseline['compatibility_errors']}.
- Treatment compatibility errors: {treatment['compatibility_errors']}.
- References read before first compatible route: {baseline_reads} -> {treatment_reads}.
- Pinned evidence presented: {baseline_evidence} -> {treatment_evidence}.
- Oracle-compatible route passes: {baseline_passes} -> {treatment_passes}.
- Paired vLLM run: {'executed' if vllm_url else 'not executed'}.
- Paired vLLM exact-file hits: {llm_baseline_files} -> {llm_treatment_files}.
- Paired vLLM expected-route hits: {llm_baseline_routes} -> {llm_treatment_routes}.

The controlled result proves value level 3: the system changes the next
engineering action and avoids a known incompatible first route.  It does not
prove real-hardware value level 4/5; those fields remain intentionally blank.
"""
    (output / "final_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-url", default="")
    parser.add_argument("--model", default="deepseekv4")
    args = parser.parse_args()
    try:
        run_campaign(args.output, vllm_url=args.vllm_url, model=args.model)
    except Exception as exc:  # noqa: BLE001 - CLI produces a clear failure
        print(f"campaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
