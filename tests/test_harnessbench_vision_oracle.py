"""Actual lossless render observations and counterexamples; never physics stepping."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.harnessbench.tasks_v2 import V02_MODEL, V03_MODEL
from benchmarks.harnessbench.vision_oracle import judge_calibration, judge_grounding, verify_camera
from rosclaw.sim.backends.mujoco import backend as mujoco_backend
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
    # Exercise the producer with an opposite inherited platform. Its manifest
    # declares the renderer actually available, not an assumption about EGL.
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    try:
        observed = runtime.observe(model, state, channels=["camera_segmentation:cam"])
    except ValueError as exc:
        if str(exc).startswith("SIM_RENDER_UNAVAILABLE"):
            pytest.fail(f"INFRASTRUCTURE_FAILURE: {exc}")
        raise
    ref = observed["values"]["camera_segmentation:cam"]["observation_manifest_ref"]
    selected = runtime.backend.store.get(ref)["renderer_backend"]
    assert selected in {"egl", "osmesa"}
    inherited = "osmesa" if selected == "egl" else "egl"
    monkeypatch.setenv("PYOPENGL_PLATFORM", inherited)
    monkeypatch.setenv("MUJOCO_GL", inherited)
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


# Linux CI contract. Only an exact missing selected SONAME is known unavailable.
# Package presence, find_library, historical tracebacks and imports are NOT proof
# of offscreen capability. These children are executed only in the renderer phase.
_RENDERER_LIBRARIES = {"egl": "libEGL.so.1", "osmesa": "libOSMesa.so.8"}
_RENDERER_MARKER = "REAL_RENDERER_CLOSED_ZERO_STEP"
_RENDERER_CAPABILITY_SCHEMA = "renderer_capability.v1"
_RENDERER_NATIVE_PROBE_CODE = r"""
import ctypes
import json
import sys

selected = sys.argv[1]
soname = {"egl": "libEGL.so.1", "osmesa": "libOSMesa.so.8"}[selected]
if not sys.platform.startswith("linux"):
    raise RuntimeError("RENDERER_CAPABILITY_UNKNOWN_PLATFORM")
try:
    native_library = ctypes.CDLL(soname)
except OSError as exc:
    # Missing dependency, permissions, symbols or a broken driver are NOT this
    # narrow absence classification; they propagate as failures.
    detail = str(exc)
    if detail != soname + ": cannot open shared object file: No such file or directory":
        raise
    print(json.dumps({
        "schema": "renderer_capability.v1", "backend": selected,
        "status": "UNAVAILABLE", "stage": "native_loader",
        "reason": "SELECTED_NATIVE_LIBRARY_ABSENT", "library": soname,
        "exception_type": "OSError", "detail": detail,
    }, sort_keys=True))
    raise SystemExit(0)
"""
_RENDERER_CONSTRUCTOR_CODE = r"""
assert os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == backend
assert hasattr(mujoco, "Renderer")
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
)
renderer = mujoco.Renderer(model, height=32, width=32)
renderer.close()
"""


def _renderer_prefix(worker):
    source = getattr(mujoco_backend, worker)
    assert "request = json.loads" in source, "PRODUCTION_PREFIX_BOUNDARY_CHANGED"
    return source.split("request = json.loads", 1)[0]


def _required_renderer_backends():
    declared = os.environ.get("ROSCLAW_REQUIRED_RENDERER_BACKENDS", "osmesa")
    required = set(declared.split(","))
    assert required <= _RENDERER_LIBRARIES.keys() and "osmesa" in required, (
        f"INVALID_REQUIRED_RENDERER_BACKENDS: {declared!r}; OSMesa cannot be downgraded"
    )
    return required


def _run_renderer_child(code, selected, inherited, label):
    assert selected in _RENDERER_LIBRARIES and inherited in _RENDERER_LIBRARIES
    env = dict(os.environ, MUJOCO_GL=inherited, PYOPENGL_PLATFORM=inherited)
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, selected],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and waits for its owned child on timeout.
        raise AssertionError(f"INFRASTRUCTURE_FAILURE: {label}: child timeout") from exc
    # Fail BEFORE interpreting any output as absence. Signals/native crashes,
    # import/constructor defects and evidence errors can never turn into skips.
    assert result.returncode == 0, (
        f"INFRASTRUCTURE_FAILURE: {label}: returncode={result.returncode}: "
        f"{result.stdout}{result.stderr}"
    )
    return result


def _probe_renderer_capability(selected):
    prefix = _renderer_prefix("_CAMERA_WORKER_CODE")
    code = (
        _RENDERER_NATIVE_PROBE_CODE
        + prefix
        + _RENDERER_CONSTRUCTOR_CODE
        + '\nprint(json.dumps({"schema": "renderer_capability.v1", '
        '"backend": backend, "status": "AVAILABLE", '
        '"stage": "offscreen_constructor", '
        '"reason": "REAL_RENDERER_CLOSED_ZERO_STEP"}, sort_keys=True))\n'
    )
    result = _run_renderer_child(code, selected, selected, f"capability/{selected}")
    # Malformed/unknown output is a hard failure, not an absence heuristic.
    capability = json.loads(result.stdout)
    assert isinstance(capability, dict), "INVALID_RENDERER_CAPABILITY_RECORD"
    capability["probe"] = {
        "worker": "_CAMERA_WORKER_CODE",
        "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "selected": selected,
        "inherited": selected,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return capability


def _renderer_capability_verdict(selected, capability, required):
    assert capability.get("schema") == _RENDERER_CAPABILITY_SCHEMA, (
        "INVALID_RENDERER_CAPABILITY_SCHEMA"
    )
    assert capability.get("backend") == selected, "RENDERER_CAPABILITY_BACKEND_MISMATCH"
    probe = capability.get("probe", {})
    assert (
        probe.get("returncode") == 0
        and probe.get("selected") == probe.get("inherited") == selected
        and probe.get("worker") == "_CAMERA_WORKER_CODE"
        and probe.get("prefix_sha256")
        == hashlib.sha256(_renderer_prefix("_CAMERA_WORKER_CODE").encode()).hexdigest()
    ), "INVALID_RENDERER_CAPABILITY_PROBE_EVIDENCE"
    if capability.get("status") == "AVAILABLE":
        assert capability.get("stage") == "offscreen_constructor", (
            "AVAILABLE_REQUIRES_REAL_OFFSCREEN_CONSTRUCTION"
        )
        assert capability.get("reason") == _RENDERER_MARKER
        return {"type": _RENDERER_CAPABILITY_SCHEMA, "verdict": "RUN", "capability": capability}
    assert capability.get("status") == "UNAVAILABLE", "RENDERER_CAPABILITY_UNKNOWN"
    soname = _RENDERER_LIBRARIES[selected]
    assert (
        capability.get("stage") == "native_loader"
        and capability.get("reason") == "SELECTED_NATIVE_LIBRARY_ABSENT"
        and capability.get("library") == soname
        and capability.get("exception_type") == "OSError"
        and capability.get("detail")
        == soname + ": cannot open shared object file: No such file or directory"
    ), "UNPROVED_RENDERER_UNAVAILABILITY"
    assert selected not in required, (
        "INFRASTRUCTURE_FAILURE: REQUIRED_RENDERER_UNAVAILABLE: "
        + json.dumps(capability, sort_keys=True)
    )
    return {
        "type": _RENDERER_CAPABILITY_SCHEMA,
        "verdict": "NOT_RUN",
        "reason": "OPTIONAL_SELECTED_NATIVE_LIBRARY_ABSENT",
        "capability": capability,
    }


def _exercise_renderer_matrix(worker, selected, inherited, capability, required):
    verdict = _renderer_capability_verdict(selected, capability, required)
    if verdict["verdict"] == "NOT_RUN":
        return verdict
    prefix = _renderer_prefix(worker)
    result = _run_renderer_child(
        prefix + _RENDERER_CONSTRUCTOR_CODE + f"\nprint({_RENDERER_MARKER!r})\n",
        selected,
        inherited,
        f"{worker}/{selected}/{inherited}",
    )
    assert result.stdout.strip() == _RENDERER_MARKER, "REAL_RENDERER_CLOSE_MARKER_MISSING"
    return {**verdict, "verdict": "PASS", "worker": worker, "inherited": inherited}


@pytest.fixture(scope="module")
def renderer_capabilities():
    # Cache per selected backend within this test process only; no stale external
    # declaration can label a backend available. Parent GL modules/env untouched.
    return {}


def _cached_renderer_capability(capabilities, selected):
    if selected not in capabilities:
        capabilities[selected] = _probe_renderer_capability(selected)
    return capabilities[selected]


def test_required_renderer_backend_offscreen_capability(renderer_capabilities, record_property):
    for selected in sorted(_required_renderer_backends()):
        capability = _cached_renderer_capability(renderer_capabilities, selected)
        verdict = _renderer_capability_verdict(selected, capability, _required_renderer_backends())
        record_property(f"renderer_capability_{selected}", json.dumps(verdict, sort_keys=True))
        assert verdict["verdict"] == "RUN"


@pytest.mark.parametrize("worker", ["_CAMERA_WORKER_CODE", "_RENDER_WORKER_CODE"])
@pytest.mark.parametrize("selected", ["egl", "osmesa"])
@pytest.mark.parametrize("inherited", ["egl", "osmesa"])
def test_real_worker_renderer_platform_matrix(
    worker, selected, inherited, renderer_capabilities, record_property
):
    # Every available combination still executes the exact production prefix
    # and a real Renderer in a fresh child, including mismatched inherited env.
    capability = _cached_renderer_capability(renderer_capabilities, selected)
    verdict = _exercise_renderer_matrix(
        worker, selected, inherited, capability, _required_renderer_backends()
    )
    evidence = json.dumps(verdict, sort_keys=True)
    record_property("renderer_capability", evidence)
    if verdict["verdict"] == "NOT_RUN":
        pytest.skip(evidence)  # Typed NOT_RUN, NOT a passing renderer matrix.
    assert verdict["verdict"] == "PASS"


# SOURCE selector: pure supplied records + mocked subprocess responses only.
# No fixture here calls any MuJoCo model/Renderer/engine API or real subprocess.
def _mock_renderer_capability(selected, status="AVAILABLE"):
    record = {
        "schema": _RENDERER_CAPABILITY_SCHEMA,
        "backend": selected,
        "status": status,
        "stage": "offscreen_constructor",
        "reason": _RENDERER_MARKER,
    }
    if status == "UNAVAILABLE":
        soname = _RENDERER_LIBRARIES[selected]
        record.update(
            stage="native_loader",
            reason="SELECTED_NATIVE_LIBRARY_ABSENT",
            library=soname,
            exception_type="OSError",
            detail=soname + ": cannot open shared object file: No such file or directory",
        )
    return record


def _mock_probe_response(monkeypatch, record, returncode=0, stderr=""):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(record), stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_renderer_capability_optional_absence_is_typed_not_run_with_exact_evidence(monkeypatch):
    calls = _mock_probe_response(monkeypatch, _mock_renderer_capability("egl", "UNAVAILABLE"))
    capability = _probe_renderer_capability("egl")
    verdict = _exercise_renderer_matrix(
        "_RENDER_WORKER_CODE", "egl", "osmesa", capability, {"osmesa"}
    )
    assert len(calls) == 1  # No matrix child for confirmed optional absence.
    assert verdict["type"] == "renderer_capability.v1"
    assert verdict["verdict"] == "NOT_RUN" and verdict["verdict"] != "PASS"
    assert verdict["reason"] == "OPTIONAL_SELECTED_NATIVE_LIBRARY_ABSENT"
    assert verdict["capability"]["library"] == "libEGL.so.1"
    assert verdict["capability"]["probe"]["returncode"] == 0
    assert "libEGL.so.1" in verdict["capability"]["probe"]["stdout"]


@pytest.mark.parametrize("status", ["UNAVAILABLE", "UNKNOWN"])
def test_renderer_capability_required_osmesa_absent_or_unknown_fails(monkeypatch, status):
    _mock_probe_response(monkeypatch, _mock_renderer_capability("osmesa", status))
    capability = _probe_renderer_capability("osmesa")
    with pytest.raises(AssertionError, match="REQUIRED_RENDERER_UNAVAILABLE|CAPABILITY_UNKNOWN"):
        _renderer_capability_verdict("osmesa", capability, {"osmesa"})


def test_renderer_capability_declared_required_egl_cannot_skip(monkeypatch):
    _mock_probe_response(monkeypatch, _mock_renderer_capability("egl", "UNAVAILABLE"))
    with pytest.raises(AssertionError, match="REQUIRED_RENDERER_UNAVAILABLE"):
        _renderer_capability_verdict("egl", _probe_renderer_capability("egl"), {"egl", "osmesa"})


@pytest.mark.parametrize(
    "detail",
    [
        "ImportError: injected available import defect",
        "RuntimeError: injected Renderer constructor defect",
        "RuntimeError: injected render defect",
        "AttributeError: 'NoneType' object has no attribute 'eglQueryString'",
    ],
)
def test_renderer_capability_available_backend_regression_fails(monkeypatch, detail):
    _mock_probe_response(monkeypatch, _mock_renderer_capability("egl"))
    capability = _probe_renderer_capability("egl")
    calls = _mock_probe_response(monkeypatch, {}, returncode=1, stderr=detail)
    with pytest.raises(AssertionError, match="INFRASTRUCTURE_FAILURE") as error:
        _exercise_renderer_matrix("_CAMERA_WORKER_CODE", "egl", "osmesa", capability, {"osmesa"})
    assert len(calls) == 1 and detail in str(error.value)


@pytest.mark.parametrize(
    "error",
    [
        "OBSERVATION_NOT_ORIGINAL_INITIAL_STATE",
        "SIM_CAMERA_STATE_INVALID",
        "MODEL_COMPILE_FAILED: bad scene",
    ],
)
def test_renderer_capability_absence_output_cannot_mask_evidence_error(monkeypatch, error):
    # Even a plausible absence record is ignored when the worker failed.
    _mock_probe_response(
        monkeypatch, _mock_renderer_capability("egl", "UNAVAILABLE"), returncode=1, stderr=error
    )
    with pytest.raises(AssertionError, match=error):
        _probe_renderer_capability("egl")


def test_renderer_capability_original_state_error_remains_evidence_failure(monkeypatch):
    test_renderer_worker_initial_state_mismatch_still_rejected_as_evidence_failure(monkeypatch)


@pytest.mark.parametrize(
    "detail",
    [
        "libGLdispatch.so.0: cannot open shared object file: No such file or directory",
        "libEGL.so.1: permission denied",
    ],
)
def test_renderer_capability_unproved_absence_never_skips(monkeypatch, detail):
    record = _mock_renderer_capability("egl", "UNAVAILABLE")
    record["detail"] = detail
    _mock_probe_response(monkeypatch, record)
    with pytest.raises(AssertionError, match="UNPROVED_RENDERER_UNAVAILABILITY"):
        _renderer_capability_verdict("egl", _probe_renderer_capability("egl"), {"osmesa"})


@pytest.mark.parametrize("worker", ["_CAMERA_WORKER_CODE", "_RENDER_WORKER_CODE"])
@pytest.mark.parametrize(
    "selected,inherited",
    [("egl", "egl"), ("egl", "osmesa"), ("osmesa", "egl"), ("osmesa", "osmesa")],
)
def test_renderer_capability_available_matrix_exact_prefix_and_parent_isolation(
    monkeypatch, worker, selected, inherited
):
    monkeypatch.setenv("MUJOCO_GL", inherited)
    monkeypatch.setenv("PYOPENGL_PLATFORM", inherited)
    parent_env = dict(os.environ)
    parent_gl_modules = {
        name: module for name, module in sys.modules.items() if name.startswith("OpenGL")
    }
    calls = _mock_probe_response(monkeypatch, _mock_renderer_capability(selected))
    capability = _probe_renderer_capability(selected)
    probe_args, probe_kwargs = calls[0]
    assert _renderer_prefix("_CAMERA_WORKER_CODE") in probe_args[0][3]
    assert _RENDERER_NATIVE_PROBE_CODE in probe_args[0][3]
    assert probe_kwargs["env"]["MUJOCO_GL"] == probe_kwargs["env"]["PYOPENGL_PLATFORM"] == selected

    def matrix_run(args, **kwargs):
        assert args[:3] == [sys.executable, "-B", "-c"] and args[-1] == selected
        assert args[3].startswith(_renderer_prefix(worker))
        assert _RENDERER_CONSTRUCTOR_CODE in args[3]
        assert kwargs["env"]["MUJOCO_GL"] == kwargs["env"]["PYOPENGL_PLATFORM"] == inherited
        assert kwargs["timeout"] == 30 and kwargs["capture_output"] and kwargs["text"]
        return SimpleNamespace(returncode=0, stdout=_RENDERER_MARKER + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", matrix_run)
    verdict = _exercise_renderer_matrix(worker, selected, inherited, capability, {"osmesa"})
    assert verdict["verdict"] == "PASS"  # Mock control flow, NOT real capability evidence.
    assert dict(os.environ) == parent_env
    assert {
        name: module for name, module in sys.modules.items() if name.startswith("OpenGL")
    } == parent_gl_modules


@pytest.mark.parametrize("failure", ["timeout", "signal"])
def test_renderer_capability_timeout_or_native_crash_fails(monkeypatch, failure):
    _mock_probe_response(monkeypatch, _mock_renderer_capability("egl"))
    capability = _probe_renderer_capability("egl")

    def failed_run(args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return SimpleNamespace(returncode=-11, stdout="", stderr="native crash")

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(AssertionError, match="INFRASTRUCTURE_FAILURE"):
        _exercise_renderer_matrix("_RENDER_WORKER_CODE", "egl", "osmesa", capability, {"osmesa"})


def test_renderer_capability_required_policy_cannot_downgrade_osmesa(monkeypatch):
    monkeypatch.setenv("ROSCLAW_REQUIRED_RENDERER_BACKENDS", "egl")
    with pytest.raises(AssertionError, match="INVALID_REQUIRED_RENDERER_BACKENDS"):
        _required_renderer_backends()


def test_renderer_capability_package_presence_is_not_offscreen_availability(monkeypatch):
    record = _mock_renderer_capability("osmesa")
    record["stage"] = "native_loader"
    _mock_probe_response(monkeypatch, record)
    with pytest.raises(AssertionError, match="AVAILABLE_REQUIRES_REAL_OFFSCREEN_CONSTRUCTION"):
        _renderer_capability_verdict("osmesa", _probe_renderer_capability("osmesa"), {"osmesa"})
