"""CLI helper for calling a registered provider capability with image input.

This module is used by ``rosclaw provider call`` and by the
``scene_risk_scan`` skill runtime handler.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.registry import ProviderRegistry
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


def _image_to_base64(image_path: Path) -> tuple[str, str]:
    data = image_path.read_bytes()
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", mime


def _build_messages(image_url: str, prompt: str) -> list[dict[str, Any]]:
    system = (
        "You are a scene safety assistant for a physical robot equipped with RealSense cameras. "
        "Analyze the provided image and return a single JSON object with these keys: "
        "scene (string), obstacles (list of strings), risks (list of {category, severity, description}), "
        "risk_score (float 0-1), executable (bool), requires_guard (bool). "
        "Do not include markdown or extra text."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def _parse_assistant_content(content: str) -> dict[str, Any]:
    """Try to parse the assistant message as JSON; fall back to wrapping raw text."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "raw": content,
            "scene": "unknown",
            "obstacles": [],
            "risks": [{"category": "parse_error", "severity": "warning", "description": "Model did not return valid JSON"}],
            "risk_score": 0.0,
            "executable": False,
            "requires_guard": True,
        }


def _normalize_vlm_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAI-compatible chat-completion response into ROSClaw result."""
    choices = raw.get("choices", [{}])
    choice = choices[0] if choices and isinstance(choices, list) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "")
    parsed = _parse_assistant_content(content)
    return {
        "scene": parsed.get("scene", "unknown"),
        "obstacles": parsed.get("obstacles", []),
        "risks": parsed.get("risks", []),
        "risk_score": float(parsed.get("risk_score", 0.0)),
        "executable": bool(parsed.get("executable", False)),
        "requires_guard": bool(parsed.get("requires_guard", True)),
        "raw_response": raw,
    }


def _get_active_practice_id(home: Path) -> str | None:
    pid_file = home / "practice" / "coordinator.pid"
    if not pid_file.exists():
        return None
    try:
        lines = pid_file.read_text(encoding="utf-8").strip().splitlines()
        return lines[1] if len(lines) > 1 else None
    except Exception:
        return None


def _record_provider_trace(home: Path, practice_id: str | None, record: dict[str, Any]) -> None:
    if not practice_id:
        return
    artifact_dir = home / "practice" / "artifacts" / practice_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "provider_trace.jsonl"
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _ensure_provider_loaded(name: str, home: Path | None = None) -> Provider:
    """Return a loaded provider, installing from ~/.rosclaw/providers if needed."""
    if home is None:
        home = Path(resolve_home())
    registry = ProviderRegistry()

    if name not in registry.list_providers():
        manifest_path = home / "providers" / f"{name}.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Provider '{name}' is not installed (manifest not found at {manifest_path})")
        manifest = ProviderManifest.from_yaml(manifest_path)
        registry.register(manifest, GenericProvider, auto_load=False)

    provider = registry.get(name)
    return provider


async def call_provider(
    provider_id: str,
    capability: str,
    image_path: str | Path,
    prompt: str | None = None,
    robot_id: str | None = None,
    task_id: str | None = None,
    timeout_ms: int | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Call a provider capability with a real image and return a normalized result.

    The result includes ``input_frame_uri``, ``latency_ms``,
    ``normalized_risk``, and ``provider_response``. It is also appended to the
    active practice session's provider trace if one exists.
    """
    if home is None:
        home = Path(resolve_home())
    image_path = Path(image_path)
    if not image_path.exists():
        return {"status": "failed", "error": f"Image not found: {image_path}"}

    prompt = prompt or "Describe the scene and identify any safety risks for a robot moving in this space."
    image_url, _ = _image_to_base64(image_path)

    provider = _ensure_provider_loaded(provider_id, home)
    manifest = provider.manifest
    model_name = manifest.model.name if manifest.model else "default"

    request_id = str(uuid.uuid4())
    request = ProviderRequest(
        request_id=request_id,
        capability=capability,
        inputs={
            "model": model_name,
            "messages": _build_messages(image_url, prompt),
            "max_tokens": 512,
            "temperature": 0.2,
        },
        context={
            "robot": robot_id or "rosclaw_default",
            "task": task_id or capability,
            "provider": provider_id,
            "image_path": str(image_path.resolve()),
        },
        constraints={"latency_ms": timeout_ms} if timeout_ms else {},
    )

    await provider.load()
    start = time.monotonic()
    try:
        response: ProviderResponse = await provider.infer(request)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc), "request_id": request_id}
    finally:
        await provider.unload()

    latency_ms = int((time.monotonic() - start) * 1000)
    raw = response.result if isinstance(response.result, dict) else {"result": response.result}
    normalized = _normalize_vlm_result(raw)

    result = {
        "status": response.status if response.status in ("ok", "degraded", "failed", "blocked") else "ok",
        "request_id": request_id,
        "provider": provider_id,
        "capability": capability,
        "model": model_name,
        "input_frame_uri": f"file://{image_path.resolve()}",
        "latency_ms": latency_ms,
        "normalized_risk": normalized["risk_score"],
        "scene": normalized["scene"],
        "obstacles": normalized["obstacles"],
        "risks": normalized["risks"],
        "executable": normalized["executable"],
        "requires_guard": normalized["requires_guard"],
        "provider_response": raw,
        "warnings": response.warnings,
        "errors": response.errors,
    }

    practice_id = _get_active_practice_id(home)
    _record_provider_trace(
        home,
        practice_id,
        {
            "timestamp": time.time(),
            "practice_id": practice_id,
            "provider": provider_id,
            "capability": capability,
            "request_id": request_id,
            "input_frame_uri": result["input_frame_uri"],
            "latency_ms": latency_ms,
            "normalized_risk": result["normalized_risk"],
            "result": normalized,
        },
    )
    return result
