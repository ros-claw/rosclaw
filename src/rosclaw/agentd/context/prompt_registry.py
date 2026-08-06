"""Prompt registry: versioned, hashed canonical prompts (总纲 §6.3).

Prompts carry ``prompt_id``, semver, content hash and the AgentLoop schema
they target. Stable-rule changes require an ADR and regression evaluation;
auto-generated patches may only ever enter a candidate area — never this
registry directory.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

_NAME_RE = re.compile(r"^(?P<prompt_id>[a-z0-9_]+)_v(?P<major>\d+)(?:\.(?P<minor>\d+))?\.md$")


@dataclass(frozen=True)
class PromptInfo:
    prompt_id: str
    version: str
    content_hash: str
    path: Path
    text: str


def _hash_text(text: str) -> str:
    return "prompt_" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt(name: str, *, prompts_dir: Path | None = None) -> PromptInfo:
    """Load e.g. ``native_agent_v1.md`` with version and hash metadata."""
    directory = prompts_dir or PROMPTS_DIR
    match = _NAME_RE.match(name)
    if not match:
        raise ValueError(f"illegal prompt file name {name!r}")
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"prompt {name!r} not found in {directory}")
    text = path.read_text(encoding="utf-8")
    minor = match.group("minor") or "0"
    return PromptInfo(
        prompt_id=match.group("prompt_id"),
        version=f"{match.group('major')}.{minor}.0",
        content_hash=_hash_text(text),
        path=path,
        text=text,
    )


def list_prompts(*, prompts_dir: Path | None = None) -> list[PromptInfo]:
    directory = prompts_dir or PROMPTS_DIR
    return [
        load_prompt(p.name, prompts_dir=directory)
        for p in sorted(directory.glob("*.md"))
        if _NAME_RE.match(p.name)
    ]
