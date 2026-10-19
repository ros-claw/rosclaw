"""Common frozen fact/episode binding for new A/B/C/D/E clients; no execution."""

import json

from benchmarks.ros_expert_ablation.rsk_pilot_protocol import allocate_pilot


def registered_episode_inputs(
    task,
    aci,
    *,
    protocol,
    root,
    ordinal,
    arm,
    facts_manifest,
    evaluation_episode,
    provider_settings,
    memory_policy,
    require_complete_originals=False,
):
    # Lazy import keeps one authoritative input schema/reader with no import cycle.
    from benchmarks.ros_expert_ablation.rsk_episode_runner import EpisodeFactsV1, _read

    base = aci.base
    allocation = next(row for row in allocate_pilot(protocol) if row["ordinal"] == ordinal)
    if allocation["arm"] != arm:
        raise ValueError("registered allocation arm differs from the injected interface")
    case = next(row for row in protocol.cases if row.case_id == allocation["case_id"])
    if (
        facts_manifest.sha256 != case.facts_manifest_sha256
        or evaluation_episode.sha256 != case.evaluation_episode_sha256
        or provider_settings.sha256 != protocol.provider_settings_sha256
        or memory_policy.sha256 != protocol.memory_policy_sha256
    ):
        raise ValueError("original allocation/facts/episode/provider hashes differ")
    facts = EpisodeFactsV1.model_validate_json(_read(root, facts_manifest))
    _read(root, evaluation_episode)
    if facts.evaluation_episode_sha256 != evaluation_episode.sha256:
        raise ValueError("registered facts do not bind the original evaluation episode")
    for ref, value in (
        (facts.case, base.case.model_dump(mode="json")),
        (facts.model, base.model.to_dict()),
        (facts.requirements, base.requirements),
        (facts.staging, base.tools.initial_staging),
        (facts.task, task),
    ):
        if json.loads(_read(root, ref)) != value:
            raise ValueError("injected interface differs from registered original facts")
    if facts.original_files is None:
        if require_complete_originals:
            raise ValueError("actual episodes require all original task file bytes in the manifest")
    elif json.loads(_read(root, facts.original_files, limit=16_000_000)) != base.tools.files:
        raise ValueError("raw task files differ from the common registered originals")
    application = getattr(aci, "registered_application_inputs", None)
    if callable(application):
        if facts.application_inputs is None:
            if require_complete_originals:
                raise ValueError(
                    "actual application inputs must be registered with the common facts"
                )
        else:
            original = json.loads(_read(root, facts.application_inputs, limit=16_000_000))
            # Compiler serialization preserves input order; bind it as well as
            # semantic values instead of silently replacing an original template.
            if json.dumps(original, separators=(",", ":"), allow_nan=False) != json.dumps(
                application(), separators=(",", ":"), allow_nan=False
            ):
                raise ValueError(
                    "application inputs/order differ from the common registered originals"
                )
    elif facts.application_inputs is not None:
        raise ValueError("registered application facts require the same application capability")
    diagnostic = getattr(aci, "registered_diagnostic_inputs", None)
    if callable(diagnostic) and facts.diagnostic_inputs is None and require_complete_originals:
        raise ValueError("actual diagnostic inputs/provider policy must be registered")
    if facts.diagnostic_inputs is not None:
        _registered_diagnostic_context(
            aci, json.loads(_read(root, facts.diagnostic_inputs, limit=16_000_000))
        )
    policy = json.loads(_read(root, memory_policy))
    if (
        type(policy) is not dict
        or set(policy) != {"schema_version", "cases"}
        or policy["schema_version"] != "rosclaw.rsk_memory_policy_manifest.v1"
        or type(policy["cases"]) is not dict
        or set(policy["cases"]) != {row.case_id for row in protocol.cases}
    ):
        raise ValueError("closed frozen per-case Memory policy manifest required")
    run_id = json.loads(_read(root, facts.run_settings)).get("run_id")
    if (
        type(run_id) is not str
        or len(run_id) != 32
        or any(c not in "0123456789abcdef" for c in run_id)
    ):
        raise ValueError("original evaluation World/run identity required")
    if require_complete_originals and run_id != (base.original_run_id or base.case.run_id):
        raise ValueError("registered run identity differs from the original current case")
    if aci.memory_query is not None and (
        policy["cases"][case.case_id] != aci.memory_query.policy
        or evaluation_episode.sha256 not in aci.memory_query.excluded_episodes
        or run_id not in aci.memory_query.excluded_runs
    ):
        raise ValueError("Memory policy differs or does not exclude the current episode/run")
    return allocation, facts


def _registered_diagnostic_context(aci, original):
    """Bind common facts without granting A/D any Expert tools or callbacks."""
    from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import DiagnosticProbePolicyV1
    from benchmarks.ros_expert_ablation.rsk_operational_aci import original_operational_graph
    from rosclaw.contracts.common import content_hash

    base = aci.base
    if (
        type(original) is not dict
        or set(original) != {"schema_version", "graph", "provider_policy"}
        or original["schema_version"] != "rosclaw.rsk_diagnostic_inputs.v1"
        or original["graph"] != original_operational_graph(base).model_dump(mode="json")
    ):
        raise ValueError("diagnostic graph differs from the registered original context")
    provider = getattr(aci, "provider", None)
    if original["provider_policy"] is None:
        if provider is not None:
            raise ValueError("registered diagnostic provider policy missing")
        return
    policy_type = DiagnosticProbePolicyV1
    if type(original["provider_policy"]) is not dict:
        raise ValueError("closed registered diagnostic policy required")
    if (
        original["provider_policy"].get("schema_version")
        == "rosclaw.rsk_nav2_diagnostic_probe_policy.v1"
    ):
        from benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_provider import (
            Nav2DiagnosticProbePolicyV1,
        )

        policy_type = Nav2DiagnosticProbePolicyV1
    policy = policy_type.model_validate_json(json.dumps(original["provider_policy"]))
    if (
        policy.source_revision != base.operator_source
        or policy.original_run_id != base.original_run_id
        or policy.original_snapshot_hash != base.case.snapshot_hash
        or policy.source_urdf_sha256 != base.case.source_urdf_sha256
        or policy.requirements_hash != content_hash("navrequirements", base.requirements)
        or policy.case_staging_hash != content_hash("acistaging", base.tools.initial_staging)
        or (provider is not None and policy != provider.policy)
    ):
        raise ValueError("diagnostic provider differs from the registered original context")
