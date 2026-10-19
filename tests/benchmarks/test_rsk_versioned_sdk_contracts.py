"""Synthetic image/version boundaries only; no Docker, ROS, model or motion."""

import copy
import hashlib
import json
import subprocess
from datetime import datetime

import pytest

from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
    RECONSTRUCTED_SDK_IMAGE,
    SDK_IMAGE,
    DiagnosticProbePolicyV1,
    DiagnosticProbePolicyV2,
    NativeDiagnosticContext,
    parse_diagnostic_probe_policy,
)
from benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_operator import (
    Nav2ObservationRecipeV1,
    Nav2ObservationRecipeV2,
    parse_nav2_observation_recipe,
)
from benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_provider import (
    Nav2DiagnosticProbePolicyV1,
    Nav2DiagnosticProbePolicyV2,
    Nav2DiagnosticProbeProvider,
    parse_nav2_diagnostic_probe_policy,
)
from benchmarks.ros_expert_ablation.rsk_nav2_original_review import _check_container_resources
from benchmarks.ros_expert_ablation.rsk_registered_inputs import _registered_diagnostic_context
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from rosclaw.connectors.ros.intelligence import build_system_model
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from tests.benchmarks.test_rsk_diagnostic_originals import originals
from tests.benchmarks.test_rsk_nav2_diagnostic_operator import operator, recipe
from tests.benchmarks.test_rsk_nav2_diagnostic_provider import provider_fixture
from tests.benchmarks.test_rsk_nav2_original_review import container
from tests.benchmarks.test_rsk_operational_aci import facade


def policy_values():
    return {
        "source_revision": "a" * 40,
        "original_run_id": "b" * 32,
        "original_snapshot_hash": "rossnap_" + "c" * 64,
        "source_urdf_sha256": "d" * 64,
        "requirements_hash": "requirements",
        "case_staging_hash": "staging",
        "execution_config_sha256": "e" * 64,
        "operations": ("ENDPOINT_QOS",),
        "minimum_wall_sec": 60,
        "observation_role": "readonly_localized_observation",
    }


@pytest.mark.parametrize("nav2", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_explicit_policy_version_pins_exact_image_and_preserves_roundtrip(nav2, version):
    values = policy_values()
    if not nav2:
        values.pop("observation_role")
        values["minimum_wall_sec"] = 8
    classes = (
        (Nav2DiagnosticProbePolicyV1, Nav2DiagnosticProbePolicyV2)
        if nav2
        else (DiagnosticProbePolicyV1, DiagnosticProbePolicyV2)
    )
    parse = parse_nav2_diagnostic_probe_policy if nav2 else parse_diagnostic_probe_policy
    original = classes[version - 1](**values)
    assert original.sdk_image == (SDK_IMAGE if version == 1 else RECONSTRUCTED_SDK_IMAGE)
    assert parse(original).model_dump_json() == original.model_dump_json()
    assert parse(original.model_dump()).model_dump_json() == original.model_dump_json()
    # A schema change never permits swapping the pinned implementation.
    for bad_image in [RECONSTRUCTED_SDK_IMAGE if version == 1 else SDK_IMAGE, "sha256:" + "f" * 64]:
        with pytest.raises(ValueError):
            parse({**original.model_dump(), "sdk_image": bad_image})
    with pytest.raises(ValueError, match="unsupported"):
        parse({**original.model_dump(), "schema_version": "unregistered.v3"})


def test_v1_recipe_keeps_exact_legacy_serialized_fields():
    original = Nav2ObservationRecipeV1(**recipe())
    expected = {
        "schema_version": "rosclaw.rsk_nav2_observation_recipe.v1",
        **recipe(),
        "seed": 101,
        "max_new_physics_worlds": 1,
        "max_goals": 0,
        "max_model_calls": 0,
    }
    assert original.model_dump(mode="json") == expected
    assert original.fixture_image == SDK_IMAGE
    assert parse_nav2_observation_recipe(json.dumps(expected)) == original
    with pytest.raises(ValueError):
        parse_nav2_observation_recipe(
            json.dumps({**expected, "sdk_image": RECONSTRUCTED_SDK_IMAGE})
        )


@pytest.mark.parametrize("image", [SDK_IMAGE, "sha256:" + "f" * 64])
def test_v2_recipe_rejects_old_or_unregistered_image(image):
    with pytest.raises(ValueError):
        Nav2ObservationRecipeV2(**recipe(), sdk_image=image)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_goals", 1),
        ("max_new_physics_worlds", 2),
        ("max_model_calls", 1),
        ("seed", True),
        ("command", "untrusted"),
    ],
)
def test_v2_recipe_preserves_closed_readonly_budgets(field, value):
    with pytest.raises(ValueError):
        parse_nav2_observation_recipe(
            json.dumps(
                {
                    **recipe(),
                    "schema_version": "rosclaw.rsk_nav2_observation_recipe.v2",
                    field: value,
                }
            )
        )


def test_v2_owner_create_and_failure_cleanup_use_the_same_explicit_image(tmp_path, monkeypatch):
    op, plan, selected = operator(tmp_path, monkeypatch)
    op.recipe = Nav2ObservationRecipeV2(**recipe())
    calls = []

    def reject(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(
        "benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_operator.subprocess.check_output",
        reject,
    )
    with pytest.raises(subprocess.CalledProcessError):
        op.collect(plan, selected, remaining_sec=90)
    assert calls[0][:2] == ["docker", "create"]
    assert RECONSTRUCTED_SDK_IMAGE in calls[0] and SDK_IMAGE not in calls[0]
    assert op.started_containers == 0 and op.calls == 1
    assert (op.last_world / "original-failure.json").is_file()


@pytest.mark.parametrize("image", [SDK_IMAGE, RECONSTRUCTED_SDK_IMAGE])
def test_v2_cleanup_stops_only_its_owned_container_with_the_selected_image(
    tmp_path, monkeypatch, image
):
    op, plan, selected = operator(tmp_path, monkeypatch)
    op.recipe = Nav2ObservationRecipeV2(**recipe())
    state = {"Id": "c" * 64, "Image": image, "State": {"Running": True, "Pid": 123}}
    stops = []

    def create(argv, **kwargs):
        assert argv[:2] == ["docker", "create"] and RECONSTRUCTED_SDK_IMAGE in argv
        state["Name"] = "/" + argv[argv.index("--name") + 1]
        return state["Id"]

    def stop(argv, **kwargs):
        assert argv == ["docker", "stop", "--timeout", "3", state["Id"]]
        stops.append(argv)
        state["State"].update(Running=False, Pid=0)

    monkeypatch.setattr(
        "benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_operator.subprocess.check_output",
        create,
    )
    monkeypatch.setattr(op, "_inspect", lambda _: copy.deepcopy(state))
    monkeypatch.setattr(
        "benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_operator.subprocess.run", stop
    )
    monkeypatch.setattr(
        "benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_operator.subprocess.Popen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic startup failure")),
    )
    with pytest.raises(OSError if image == RECONSTRUCTED_SDK_IMAGE else ValueError):
        op.collect(plan, selected, remaining_sec=90)
    assert len(stops) == (1 if image == RECONSTRUCTED_SDK_IMAGE else 0)
    assert (op.last_world / "original-failure.json").is_file()


@pytest.mark.parametrize("actual", [SDK_IMAGE, RECONSTRUCTED_SDK_IMAGE])
@pytest.mark.parametrize("expected", [SDK_IMAGE, RECONSTRUCTED_SDK_IMAGE])
def test_original_container_must_match_selected_contract_without_fallback(actual, expected):
    value = container()
    value["Image"] = actual
    if actual == expected:
        _check_container_resources(value, sdk_image=expected)
    else:
        with pytest.raises(ValueError, match="fixed isolated"):
            _check_container_resources(value, sdk_image=expected)
    if actual == RECONSTRUCTED_SDK_IMAGE:
        with pytest.raises(ValueError):
            _check_container_resources(value)  # Historical default remains V1.


def test_original_reviewer_cannot_be_given_arbitrary_image():
    with pytest.raises(ValueError, match="unsupported"):
        _check_container_resources(container(), sdk_image="sha256:" + "f" * 64)


@pytest.mark.parametrize("image", [SDK_IMAGE, RECONSTRUCTED_SDK_IMAGE])
def test_native_v2_reviews_actual_image_and_rejects_rehashed_legacy_substitution(
    tmp_path, monkeypatch, image
):
    provider, reference, experiment, selected, aci = originals(tmp_path, monkeypatch, None)
    values = provider.policy.model_dump()
    values.update(
        schema_version="rosclaw.rsk_diagnostic_probe_policy.v2", sdk_image=RECONSTRUCTED_SDK_IMAGE
    )
    provider.policy = DiagnosticProbePolicyV2.model_validate(values)
    manifest_path = tmp_path / reference.path
    manifest = json.loads(manifest_path.read_bytes())
    for key in ["container_before", "container_after"]:
        path = tmp_path / manifest[key]["path"]
        value = json.loads(path.read_bytes())
        value["Image"] = image
        path.write_text(json.dumps(value))
        manifest[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    reference = reference.model_copy(
        update={"sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    )
    monkeypatch.setattr(
        "benchmarks.ros_expert_ablation.rsk_diagnostic_provider.subprocess.check_output",
        lambda *a, **k: json.dumps([value]),
    )
    if image == SDK_IMAGE:
        with pytest.raises(ValueError, match="pinned isolated"):
            provider.review_originals(
                reference, experiment, selected, aci.base, original_graph=aci._graph()
            )
    else:
        native = json.loads((tmp_path / manifest["native_before"]["path"]).read_bytes())
        model = build_system_model(
            RosGraphSnapshot.from_dict(
                {
                    "ros_version": "ros2",
                    "distro": "jazzy",
                    "endpoint": "fixture://synthetic",
                    "captured_at": native["captured_at"],
                    **native["graph"],
                }
            ),
            native=native,
        )
        base = NativeDiagnosticContext(model, "b" * 40, "a" * 32)
        graph = compile_operational_graph(
            model,
            source_ref="synthetic",
            now=datetime.fromisoformat(native["captured_at"]),
            evidence_domain="SEALED_REPLAY",
        )
        experiment = plan_diagnostic_probes(
            graph, issue_ids=[f.id for f in graph.findings], budget_sec=20, max_probes=1
        )
        selected = next(p for p in experiment.probes if p.request.operation == "ENDPOINT_QOS")
        result = provider.review_originals(
            reference, experiment, selected, base, original_graph=graph
        )
        assert result["current_authority"] == "NONE"
    assert provider.operator.calls == aci.base.probe_count == 0


def upgrade(provider):
    values = provider.policy.model_dump()
    values.update(
        schema_version="rosclaw.rsk_nav2_diagnostic_probe_policy.v2",
        sdk_image=RECONSTRUCTED_SDK_IMAGE,
    )
    return Nav2DiagnosticProbePolicyV2.model_validate(values)


def test_policy_recipe_image_mismatch_is_rejected_before_dispatch(tmp_path, monkeypatch):
    legacy, _ = provider_fixture(tmp_path, monkeypatch, facade())
    with pytest.raises(ValueError, match="operator/recipe"):
        Nav2DiagnosticProbeProvider(
            upgrade(legacy),
            operator=legacy.operator,
            worktree=legacy.worktree,
            evidence_root=tmp_path,
        )
    assert legacy.operator.calls == legacy.operator.started_containers == 0


def test_registered_v2_inputs_bind_and_detect_recipe_swaps_before_budget(tmp_path, monkeypatch):
    original = facade()
    legacy, _ = provider_fixture(tmp_path, monkeypatch, original)
    values = legacy.operator.recipe.model_dump()
    values["schema_version"] = "rosclaw.rsk_nav2_observation_recipe.v2"
    legacy.operator.recipe = Nav2ObservationRecipeV2.model_validate(values)
    provider = Nav2DiagnosticProbeProvider(
        upgrade(legacy), operator=legacy.operator, worktree=legacy.worktree, evidence_root=tmp_path
    )
    provider.check_context(original.base)
    registered = copy.deepcopy(original.registered_diagnostic_inputs())
    registered["provider_policy"] = provider.policy.model_dump(mode="json")
    original.provider = provider
    _registered_diagnostic_context(original, registered)
    # Valid hashes/roles/context cannot hide a cross-image recipe substitution.
    provider.operator.recipe = Nav2ObservationRecipeV1(**recipe())
    with pytest.raises(ValueError, match="binding changed"):
        provider.check_context(original.base)
    assert (
        provider.operator.calls
        == provider.operator.started_containers
        == original.base.probe_count
        == 0
    )
