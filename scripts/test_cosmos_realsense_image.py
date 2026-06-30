#!/usr/bin/env python3
"""Send a real RealSense image to the LAN Cosmos/vLLM endpoint.

Usage:
    ./scripts/test_cosmos_realsense_image.py --image /tmp/d405_color.jpg \
        --endpoint http://192.168.1.105:8009/v1/chat/completions
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
        # Drop ```json and trailing ```
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


def _normalize_result(body: dict[str, Any]) -> dict[str, Any]:
    choice = body.get("choices", [{}])[0]
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
        "model": body.get("model", "unknown"),
        "usage": body.get("usage", {}),
        "raw_response": body,
    }


def infer(image_path: Path, endpoint: str, model: str, prompt: str, timeout_sec: float = 60.0) -> dict[str, Any]:
    image_url, _ = _image_to_base64(image_path)
    payload = {
        "model": model,
        "messages": _build_messages(image_url, prompt),
        "max_tokens": 512,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = {"error": f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')}"}
        return {"status": "failed", **body, "latency_ms": (time.monotonic() - start) * 1000}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc), "latency_ms": (time.monotonic() - start) * 1000}

    latency_ms = (time.monotonic() - start) * 1000
    result = _normalize_result(body)
    result["status"] = "ok"
    result["latency_ms"] = latency_ms
    result["input_frame_uri"] = f"file://{image_path.resolve()}"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test Cosmos real-image inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--endpoint", default="http://192.168.1.105:8009/v1/chat/completions", help="vLLM endpoint")
    parser.add_argument("--model", default="qwen3-vl-2b-lan", help="Model name")
    parser.add_argument("--prompt", default="Describe the scene and identify any safety risks for a robot moving in this space.", help="User prompt")
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    result = infer(image_path, args.endpoint, args.model, args.prompt, args.timeout)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
