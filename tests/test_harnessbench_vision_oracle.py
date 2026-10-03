"""Actual lossless render observations and counterexamples; never physics stepping."""

import copy

import numpy as np
import pytest

from benchmarks.harnessbench.tasks_v2 import V02_MODEL, V03_MODEL
from benchmarks.harnessbench.vision_oracle import judge_calibration, judge_grounding, verify_camera
from rosclaw.sim.backends.mujoco.backend import MujocoBackend


@pytest.fixture(scope="module")
def observations(tmp_path_factory):
    backend = MujocoBackend(tmp_path_factory.mktemp("vision_oracle"))
    ground = None
    calibration = {"consistent": True, "camera_evidence": []}
    for xml, cameras, body in (
        (V02_MODEL, ["cam"], "blue_box"),
        (V03_MODEL, ["cam_a", "cam_b"], "cube"),
    ):
        model = backend.load_model_xml(xml, source={"kind": "fixture"}).model_ref
        state = backend.initial_state_v2(model)
        for camera in cameras:
            try:
                observed = backend.observe(
                    model,
                    state,
                    [
                        "camera_segmentation:" + camera,
                        "camera_depth:" + camera,
                    ],
                ).values
            except ValueError as exc:
                if str(exc).startswith("SIM_RENDER_UNAVAILABLE"):
                    pytest.skip(str(exc))
                raise
            seg_ref = observed["camera_segmentation:" + camera]["observation_manifest_ref"]
            depth_ref = observed["camera_depth:" + camera]["observation_manifest_ref"]
            truth = verify_camera(
                backend, seg_ref, source_xml=xml, camera=camera, body=body, kind="segmentation"
            )
            if body == "blue_box":
                y, x = np.argwhere(truth["target_mask"])[0]
                ground = {
                    "segment_label": truth["geom_id"],
                    "segment_object_type": truth["object_type"],
                    "pixel": [int(x), int(y)],
                    "segmentation_manifest_ref": seg_ref,
                }
            else:
                calibration["camera_evidence"].append(
                    {
                        "camera": camera,
                        "segmentation_manifest_ref": seg_ref,
                        "depth_manifest_ref": depth_ref,
                        "projected_center_px": truth["projected_center_px"],
                    }
                )
    return backend, ground, calibration


def test_actual_lossless_pixels_and_two_camera_geometry_pass(observations):
    backend, grounding, calibration = observations
    assert judge_grounding(backend, grounding, V02_MODEL)["pixel_count"] > 4
    checked = judge_calibration(backend, calibration, V03_MODEL)
    assert len(checked["camera_checks"]) == 2
    assert checked["physics_steps_by_oracle"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("segment_label", "blue"),
        ("segment_label", 999),
        ("segment_label", True),
        ("segment_object_type", 0),
        ("pixel", [0, 0]),
    ],
)
def test_grounding_wrong_label_or_non_target_pixel_rejected(observations, key, value):
    backend, grounding, _ = observations
    with pytest.raises(ValueError, match="SEGMENT"):
        judge_grounding(backend, {**grounding, key: value}, V02_MODEL)


@pytest.mark.parametrize("mutation", ["false", "repeat", "projection", "swapped"])
def test_calibration_false_duplicate_camera_wrong_projection_or_channels_rejected(
    observations, mutation
):
    backend, _, calibration = observations
    mutated = copy.deepcopy(calibration)
    if mutation == "false":
        mutated["consistent"] = False
    elif mutation == "repeat":
        mutated["camera_evidence"][1] = copy.deepcopy(mutated["camera_evidence"][0])
    elif mutation == "projection":
        mutated["camera_evidence"][0]["projected_center_px"][0] += 10
    else:
        item = mutated["camera_evidence"][0]
        item["depth_manifest_ref"] = item["segmentation_manifest_ref"]
    with pytest.raises(ValueError):
        judge_calibration(backend, mutated, V03_MODEL)


@pytest.mark.parametrize("mutation", ["calibration", "raw", "state", "digest"])
def test_camera_manifest_counterexamples_fail_closed(observations, mutation):
    backend, grounding, _ = observations
    altered = copy.deepcopy(backend.store.get(grounding["segmentation_manifest_ref"]))
    if mutation == "calibration":
        altered["extrinsics"]["pos"][0] += 0.1
    elif mutation == "raw":
        import io

        blob = io.BytesIO()
        np.save(blob, np.zeros((480, 640, 2), dtype=np.int32), allow_pickle=False)
        altered["raw_artifact_ref"] = backend.store.put("renders", blob.getvalue())
    elif mutation == "state":
        state = backend.store.get(altered["state_ref"])
        state["model_ref"] = "simmdl_" + "0" * 16
        altered["state_ref"] = backend.store.put("states", state)
    else:
        altered["model_digest"] = "sha256:invalid"
    ref = backend.store.put("renders", altered)
    with pytest.raises(ValueError):
        judge_grounding(backend, {**grounding, "segmentation_manifest_ref": ref}, V02_MODEL)


def test_public_camera_pins_renderer_despite_inherited_opengl_platform(tmp_path, monkeypatch):
    from rosclaw.sim.runtime import SimulationRuntime

    (tmp_path / "scene.xml").write_text(V02_MODEL)
    runtime = SimulationRuntime(tmp_path)
    model = runtime.load_model("scene.xml")["model_ref"]
    state = runtime.snapshot(model)["state_ref"]
    # Producer's declared renderer is EGL; the operator shell may select OSMesa.
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    observed = runtime.observe(model, state, channels=["camera_segmentation:cam"])
    ref = observed["values"]["camera_segmentation:cam"]["observation_manifest_ref"]
    assert runtime.backend.store.get(ref)["renderer_backend"] == "egl"
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    truth = verify_camera(
        runtime.backend,
        ref,
        source_xml=V02_MODEL,
        camera="cam",
        body="blue_box",
        kind="segmentation",
    )
    assert truth["pixel_count"] > 4


def test_declared_renderer_unavailable_never_falls_back_or_blames_agent(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks.harnessbench import oracle, vision_oracle

    (tmp_path / "answer.json").write_text("{}")
    requests = []

    def unavailable(*args, **kwargs):
        requests.append(kwargs["env"])
        return SimpleNamespace(returncode=1, stderr=b"EGL unavailable")

    monkeypatch.setattr(vision_oracle.subprocess, "run", unavailable)

    def judge(*_):
        return vision_oracle._independent_pixels(V02_MODEL, np.zeros(1), "cam", "blue_box", "egl")

    verdict = oracle._vision_verdict(tmp_path, judge, V02_MODEL)
    assert len(requests) == 1
    assert requests[0]["MUJOCO_GL"] == requests[0]["PYOPENGL_PLATFORM"] == "egl"
    assert verdict["verification_status"] == "INFRASTRUCTURE_FAILURE"
    assert verdict["false_success"] is False and verdict["verified_success"] is False


def test_renderer_worker_initial_state_mismatch_still_rejected_as_evidence_failure(monkeypatch):
    from types import SimpleNamespace

    from benchmarks.harnessbench import vision_oracle

    monkeypatch.setattr(
        vision_oracle.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stderr=b"OBSERVATION_NOT_ORIGINAL_INITIAL_STATE"
        ),
    )
    with pytest.raises(ValueError, match="OBSERVATION_NOT_ORIGINAL_INITIAL_STATE") as error:
        vision_oracle._independent_pixels(V02_MODEL, np.zeros(1), "cam", "blue_box", "egl")
    assert not isinstance(error.value, vision_oracle.CameraRendererUnavailableError)
