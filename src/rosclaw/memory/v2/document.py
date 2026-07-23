"""Multilingual memory document builder + exact-entity extraction (数据库优化v3 §6).

A MemoryItem's raw fields are NEVER mutated; the builder derives a
``MemoryDocument`` with parallel ZH / EN / CANONICAL sections plus alias
expansion from ``resources/robotics_lexicon.yaml``.  The combined text
feeds BM25 and the embedder; the alias list feeds lexical scoring; the
``exact_terms`` feed hard metadata constraints (§6.3 — an embedding must
never be allowed to treat ``middle`` and ``thumb_rot`` as synonyms).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_LEXICON_PATH = Path(__file__).parent / "resources" / "robotics_lexicon.yaml"


@dataclass(frozen=True)
class MemoryDocument:
    zh: str
    en: str
    canonical: str
    combined: str
    aliases: list[str] = field(default_factory=list)
    exact_terms: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, Any]:
    with _LEXICON_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def lexicon_terms() -> dict[str, dict[str, list[str]]]:
    return _lexicon().get("terms", {})


def exact_term_config() -> dict[str, Any]:
    return _lexicon().get("exact_terms", {})


def aliases_for(canonical: str) -> list[str]:
    """All zh+en aliases for a canonical term (including the term itself)."""
    entry = lexicon_terms().get(canonical)
    if not entry:
        return [canonical] if canonical else []
    out = [canonical]
    out.extend(entry.get("zh") or [])
    out.extend(entry.get("en") or [])
    return out


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*|[一-鿿]+")


def extract_exact_terms(text: str) -> dict[str, list[str]]:
    """Pull hard entities out of free text (§6.3).

    Returns dimension -> matched CANONICAL values, e.g.
    ``{"joints": ["middle"], "error_codes": ["EIO"], "hands": ["left"]}``.
    Both the canonical term and every zh/en alias match (``中指`` →
    ``middle``, ``关节未到位`` → ``joint_not_reached``).  Word-boundary
    matching for latin terms: ``_`` is a word character, so ``\bthumb\b``
    never matches inside ``thumb_rot`` and ``\bright\b`` never matches
    inside ``copyright`` — joints can never be blurred together.
    """
    if not text:
        return {}
    cfg = exact_term_config()
    found: dict[str, list[str]] = {}
    lowered = text.lower()

    def _present(term: str) -> bool:
        # CJK aliases have no word boundaries in running text; latin
        # terms must respect \b so substrings never false-positive.
        if any("一" <= ch <= "鿿" for ch in term):
            return term in text
        return re.search(rf"\b{re.escape(term.lower())}\b", lowered) is not None

    terms = lexicon_terms()
    for dim in ("joints", "gestures", "hands", "devices", "failure_types"):
        values = cfg.get(dim) or []
        hits: list[str] = []
        for value in values:
            entry = terms.get(str(value)) or {}
            candidates = [str(value)]
            candidates.extend(entry.get("zh") or [])
            candidates.extend(entry.get("en") or [])
            if dim == "hands":
                # bare "left"/"right" are too ambiguous in English
                # ("you left", "right?") — only the explicit hand forms
                # (左手/右手/left hand/right hand) count as constraints.
                candidates = [c for c in candidates if c != str(value)]
            if any(_present(alias) for alias in candidates):
                hits.append(str(value))
        # CJK aliases nest (拇指 ⊂ 拇指根关节): when the more specific
        # thumb_rot matched, a bare-thumb hit is an artifact of substring
        # matching, so thumb is dropped (same mutual-exclusion the latin
        # \b path gives for free).  A query genuinely naming both joints
        # is far rarer than the 拇指根 phrasing of thumb_rot alone.
        if dim == "joints" and "thumb_rot" in hits and "thumb" in hits:
            hits.remove("thumb")
        if hits:
            found[dim] = hits
    for pattern in cfg.get("error_codes") or []:
        if re.search(pattern, text):
            found.setdefault("error_codes", [])
            cleaned = pattern.replace("\\b", "")
            if cleaned not in found["error_codes"]:
                found["error_codes"].append(cleaned)
    return found


class MultilingualMemoryDocumentBuilder:
    """Builds ZH/EN/CANONICAL/ALIASES documents from structured fields."""

    def build(
        self,
        *,
        zh: str,
        en: str,
        canonical: str,
        extra_aliases: list[str] | None = None,
        exact_terms: list[str] | None = None,
    ) -> MemoryDocument:
        alias_set: list[str] = []
        for term in extra_aliases or []:
            for alias in aliases_for(term):
                if alias not in alias_set:
                    alias_set.append(alias)
        exact = exact_terms or []
        sections = []
        if zh.strip():
            sections.append(f"[ZH]\n{zh.strip()}")
        if en.strip():
            sections.append(f"[EN]\n{en.strip()}")
        if canonical.strip():
            sections.append(f"[CANONICAL]\n{canonical.strip()}")
        if alias_set:
            sections.append("[ALIASES]\n" + " ".join(alias_set))
        combined = "\n\n".join(sections)
        return MemoryDocument(
            zh=zh.strip(),
            en=en.strip(),
            canonical=canonical.strip(),
            combined=combined,
            aliases=alias_set,
            exact_terms=exact,
        )

    def build_failure(
        self,
        *,
        hand: str,
        joint: str | None,
        gesture: str | None,
        failure_type: str,
        round_index: int | None = None,
        temperature_c: float | None = None,
        robot: str = "RH56",
    ) -> MemoryDocument:
        """Canonical bilingual document for a joint_not_reached-style failure."""
        hand_zh = {"left": "左手", "right": "右手"}.get(hand, "未知手别")
        joint_zh = _zh_alias(joint) if joint else None
        gesture_canonical = _canonical_gesture(gesture)
        gesture_zh = _zh_alias(gesture_canonical) if gesture_canonical else None
        joint_en = joint or "unknown joint"
        gesture_en = gesture_canonical or "unknown gesture"

        zh = f"{robot} {hand_zh}执行{gesture_zh or '手势'}时，{joint_zh or '有关关节'}未到达目标位置。"
        if round_index is not None:
            zh += f"失败发生在第 {round_index} 回合。"
        if temperature_c is not None:
            zh += f"当时温度 {temperature_c:.0f}°C。"

        if joint:
            en = (
                f"{robot} {hand} hand failed to reach the target position on "
                f"{joint_en} while executing {gesture_en}."
            )
        else:
            # Honest non-attribution: the session never recorded which
            # joint failed — do not invent one, do not print "unknown
            # joint" as if it were a joint name.
            en = (
                f"{robot} {hand} hand failed to reach the target position "
                f"while executing {gesture_en} (failing joint not recorded)."
            )
        if round_index is not None:
            en += f" Failure occurred at round {round_index}."
        if temperature_c is not None:
            en += f" Temperature was {temperature_c:.0f}°C."

        canonical_parts = [
            f"robot={robot}",
            f"hand={hand}",
            f"joint={joint or 'unattributed'}",
            f"gesture={gesture_en}",
            f"failure={failure_type}",
        ]
        if round_index is not None:
            canonical_parts.append(f"round={round_index}")
        if temperature_c is not None:
            canonical_parts.append(f"temperature_c={temperature_c:.0f}")

        alias_terms = [failure_type]
        if joint:
            alias_terms.append(joint)
        if gesture_canonical:
            alias_terms.append(gesture_canonical)
        exact = [v for v in (joint, gesture_canonical, failure_type) if v]
        if hand in ("left", "right"):
            exact.append(hand)
        return self.build(
            zh=zh,
            en=en,
            canonical=" ".join(canonical_parts),
            extra_aliases=alias_terms,
            exact_terms=exact,
        )


def _zh_alias(canonical: str | None) -> str | None:
    if not canonical:
        return None
    entry = lexicon_terms().get(canonical)
    if entry and entry.get("zh"):
        return entry["zh"][0]
    return canonical


def _canonical_gesture(gesture: str | None) -> str | None:
    if not gesture:
        return None
    for prefix in ("left_", "right_"):
        if gesture.startswith(prefix):
            return gesture.removeprefix(prefix)
    return gesture
