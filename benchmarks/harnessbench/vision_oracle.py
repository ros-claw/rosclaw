"""Independent camera pixel/geometry checks, without physics steps or producer imports.

Known static benchmark scenes have no assets or moving bodies. A separate native
MuJoCo renderer recreates the full initial state, raw segmentation and metric
depth. PNG previews and model-authored text never substitute for those arrays.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.harnessbench.dynamic_oracle import same_compiled_body

_WORKER = r"""
import json, math, sys
import mujoco
import numpy as np
r = json.load(open(sys.argv[1]))
m = mujoco.MjModel.from_xml_string(r["xml"])
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
spec = mujoco.mjtState.mjSTATE_INTEGRATION
v = np.zeros(mujoco.mj_stateSize(m, spec))
mujoco.mj_getState(m, d, v, spec)
if not np.array_equal(v, np.asarray(r["vector"], dtype=np.float64)):
    raise ValueError("OBSERVATION_NOT_ORIGINAL_INITIAL_STATE")
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, r["camera"])
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r["body"])
if cid < 0 or bid < 0:
    raise ValueError("UNKNOWN_CAMERA_OR_TARGET")
ids = np.flatnonzero(m.geom_bodyid == bid)
if len(ids) != 1:
    raise ValueError("BENCH_TARGET_GEOM_AMBIGUOUS")
gid = int(ids[0])
rotation = d.cam_xmat[cid].reshape(3, 3)
local = rotation.T @ (d.geom_xpos[gid] - d.cam_xpos[cid])
focal = 480 / (2 * math.tan(math.radians(float(m.cam_fovy[cid])) / 2))
if local[2] >= 0:
    raise ValueError("TARGET_BEHIND_CAMERA")
pixel = [320 + focal * local[0] / -local[2], 240 - focal * local[1] / -local[2]]
renderer = mujoco.Renderer(m, 480, 640)
renderer.update_scene(d, camera=r["camera"])
renderer.enable_segmentation_rendering()
seg = np.asarray(renderer.render(), dtype=np.int32).copy()
renderer.disable_segmentation_rendering()
renderer.enable_depth_rendering()
depth = np.asarray(renderer.render(), dtype=np.float32).copy()
renderer.close()
np.savez(r["out"], segmentation=seg, depth=depth)
result = {"geom_id": gid, "object_type": int(mujoco.mjtObj.mjOBJ_GEOM),
          "projected_center_px": pixel, "focal_px": focal,
          "camera_pos": d.cam_xpos[cid].tolist(), "camera_mat": d.cam_xmat[cid].tolist()}
json.dump(result, open(r["metadata"], "w"), allow_nan=False)
"""


class CameraRendererUnavailableError(ValueError):
    """Verifier infrastructure failure, without a model success/failure claim."""


def _independent_pixels(xml, vector, camera, body, renderer):
    """Pin the declared GL backend in its own process; never compare across renderers."""
    with tempfile.TemporaryDirectory(prefix="rosclaw_vision_oracle_") as temporary:
        root = Path(temporary)
        request = {
            "xml": xml,
            "vector": vector.tolist(),
            "camera": camera,
            "body": body,
            "out": str(root / "pixels.npz"),
            "metadata": str(root / "metadata.json"),
        }
        path = root / "request.json"
        path.write_text(json.dumps(request, allow_nan=False))
        if renderer not in {"egl", "osmesa"}:
            raise ValueError("CAMERA_RENDERER_BINDING_INVALID")
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _WORKER, str(path)],
                env={**os.environ, "MUJOCO_GL": renderer, "PYOPENGL_PLATFORM": renderer},
                capture_output=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise CameraRendererUnavailableError(
                "INDEPENDENT_CAMERA_RENDERER_UNAVAILABLE:" + renderer
            ) from exc
        if completed.returncode != 0:
            for reason in (
                "OBSERVATION_NOT_ORIGINAL_INITIAL_STATE",
                "UNKNOWN_CAMERA_OR_TARGET",
                "BENCH_TARGET_GEOM_AMBIGUOUS",
                "TARGET_BEHIND_CAMERA",
            ):
                if reason.encode() in completed.stderr:
                    raise ValueError(reason)
            raise CameraRendererUnavailableError(
                "INDEPENDENT_CAMERA_RENDERER_UNAVAILABLE:" + renderer
            )
        with np.load(root / "pixels.npz", allow_pickle=False) as arrays:
            pixels = {key: arrays[key].copy() for key in arrays.files}
        return pixels, json.loads((root / "metadata.json").read_text())


def verify_camera(backend, manifest_ref, *, source_xml, camera, body, kind):
    """Bind typed manifest, original compiled model/state, raw arrays and calibration."""
    manifest = backend.store.get(manifest_ref)
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value
        for key, value in {
            "kind": "camera_observation",
            "schema_version": "rosclaw.sim.camera.v2",
            "channel": f"camera_{kind}:{camera}",
            "camera": camera,
            "width": 640,
            "height": 480,
            "raw_units": "categorical" if kind == "segmentation" else "metre",
        }.items()
    ):
        raise ValueError("CAMERA_MANIFEST_INVALID")
    model_ref, state_ref = manifest["model_ref"], manifest["state_ref"]
    model = backend.store.get(model_ref)
    state = backend.store.get(state_ref)
    if not isinstance(model, dict) or not isinstance(state, dict):
        raise ValueError("CAMERA_MODEL_OR_STATE_MANIFEST_INVALID")
    if model.get("assets"):
        raise ValueError("CAMERA_FIXTURE_UNEXPECTED_ASSETS")
    identity = {"xml": hashlib.sha256(model["mjcf_xml"].encode()).hexdigest(), "assets": []}
    model_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    if manifest.get("model_digest") != model_digest:
        raise ValueError("CAMERA_MODEL_DIGEST_INVALID")
    channels = ["object_id", "object_type"] if kind == "segmentation" else ["camera_forward_depth"]
    if manifest.get("raw_channels") != channels:
        raise ValueError("CAMERA_RAW_CHANNEL_SEMANTICS_INVALID")
    if not same_compiled_body(source_xml, model["mjcf_xml"]):
        raise ValueError("CAMERA_MODEL_CHANGED")
    if (
        state.get("kind") != "state_snapshot_v2"
        or state.get("fidelity") != "FULL_INTEGRATION"
        or state.get("model_ref") != model_ref
        or state.get("model_digest") != manifest.get("model_digest")
    ):
        raise ValueError("CAMERA_STATE_BINDING_INVALID")
    blob = backend.store.get(state["state_vector_ref"])
    if (
        not isinstance(blob, bytes)
        or state.get("state_digest") != "sha256:" + hashlib.sha256(blob).hexdigest()
    ):
        raise ValueError("CAMERA_STATE_DIGEST_INVALID")
    import mujoco

    if state.get("state_spec_value") != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("CAMERA_STATE_SPEC_INVALID")
    pixels, metadata = _independent_pixels(
        source_xml,
        np.frombuffer(blob, dtype=np.float64),
        camera,
        body,
        manifest.get("renderer_backend"),
    )
    raw = np.load(io.BytesIO(backend.store.get(manifest["raw_artifact_ref"])), allow_pickle=False)
    expected = pixels[kind]
    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != expected.dtype
        or raw.shape != expected.shape
        or not (
            np.array_equal(raw, expected)
            if kind == "segmentation"
            else np.allclose(raw, expected, rtol=1e-6, atol=1e-6)
        )
    ):
        raise ValueError("CAMERA_RAW_PIXELS_MISMATCH")
    if not (
        np.allclose(manifest["intrinsics"]["focal_px"], metadata["focal_px"], rtol=0, atol=1e-9)
        and manifest["intrinsics"]["principal_point"] == [320, 240]
        and np.allclose(manifest["extrinsics"]["pos"], metadata["camera_pos"], rtol=0, atol=1e-12)
        and np.allclose(manifest["extrinsics"]["mat"], metadata["camera_mat"], rtol=0, atol=1e-12)
    ):
        raise ValueError("CAMERA_CALIBRATION_MISMATCH")
    target = np.all(
        pixels["segmentation"] == [metadata["geom_id"], metadata["object_type"]], axis=2
    )
    if np.count_nonzero(target) < 4:
        raise ValueError("TARGET_NOT_VISIBLE")
    return {
        **metadata,
        "state_ref": state_ref,
        "model_ref": model_ref,
        "target_mask": target,
        "pixel_count": int(np.count_nonzero(target)),
        "physics_steps_by_oracle": 0,
    }


def judge_grounding(backend, answer, source_xml):
    result = verify_camera(
        backend,
        answer["segmentation_manifest_ref"],
        source_xml=source_xml,
        camera="cam",
        body="blue_box",
        kind="segmentation",
    )
    label, object_type, pixel = (
        answer["segment_label"],
        answer["segment_object_type"],
        answer["pixel"],
    )
    if (
        type(label) is not int
        or label != result["geom_id"]
        or type(object_type) is not int
        or object_type != result["object_type"]
    ):
        raise ValueError("SEGMENT_LABEL_WRONG")
    if not isinstance(pixel, list) or len(pixel) != 2 or any(type(p) is not int for p in pixel):
        raise ValueError("SEGMENT_PIXEL_INVALID")
    x, y = pixel
    if not (0 <= x < 640 and 0 <= y < 480) or not result["target_mask"][y, x]:
        raise ValueError("SEGMENT_PIXEL_WRONG")
    result.pop("target_mask")
    return result


def judge_calibration(backend, answer, source_xml):
    if answer.get("consistent") is not True:
        raise ValueError("STATIC_CAMERA_CONSISTENCY_CLAIM_WRONG")
    evidence = answer["camera_evidence"]
    if (
        not isinstance(evidence, list)
        or len(evidence) != 2
        or {item["camera"] for item in evidence} != {"cam_a", "cam_b"}
    ):
        raise ValueError("DUAL_CAMERA_EVIDENCE_REQUIRED")
    checks = []
    for item in evidence:
        results = [
            verify_camera(
                backend,
                item[kind + "_manifest_ref"],
                source_xml=source_xml,
                camera=item["camera"],
                body="cube",
                kind=kind,
            )
            for kind in ("segmentation", "depth")
        ]
        if any(results[0][key] != results[1][key] for key in ("state_ref", "model_ref")):
            raise ValueError("CAMERA_CHANNEL_STATE_MISMATCH")
        points = item["projected_center_px"]
        if not isinstance(points, list) or any(type(p) not in (int, float) for p in points):
            raise ValueError("CAMERA_PROJECTION_INVALID")
        projected = np.asarray(points, dtype=float)
        if (
            projected.shape != (2,)
            or not np.isfinite(projected).all()
            or np.max(np.abs(projected - results[0]["projected_center_px"])) > 0.5
        ):
            raise ValueError("CAMERA_PROJECTION_WRONG")
        results[0].pop("target_mask")
        checks.append(results[0])
    if (
        checks[0]["state_ref"] != checks[1]["state_ref"]
        or checks[0]["model_ref"] != checks[1]["model_ref"]
    ):
        raise ValueError("DUAL_CAMERA_WORLD_STATE_MISMATCH")
    return {"camera_checks": checks, "physics_steps_by_oracle": 0}
