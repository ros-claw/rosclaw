"""Render README capability matrices from the canonical product status."""

from __future__ import annotations

from typing import Any

from rosclaw.product.status import iter_matrix_entries

START_MARKER = "<!-- product-status:start -->"
END_MARKER = "<!-- product-status:end -->"

CLAIM_LABELS = {
    "component_system_verified": {
        "en": "Component/system verified",
        "zh": "组件与系统已验证",
    },
    "component_verified": {"en": "Component verified", "zh": "组件已验证"},
    "developer_observed_revalidation_pending": {
        "en": "Developer-observed; revalidation pending",
        "zh": "开发者已观察；独立复验中",
    },
    "experimental": {"en": "Experimental", "zh": "实验性"},
    "fixture_only": {"en": "Fixture only", "zh": "仅 Fixture"},
    "not_verified": {"en": "Not verified", "zh": "未验证"},
    "revalidation_pending": {"en": "Revalidation pending", "zh": "复验中"},
    "simulation_verified": {"en": "Simulation verified", "zh": "仿真已验证"},
}


def render_readme_matrix(status: dict[str, Any], language: str) -> str:
    """Render the managed Markdown table for one README language."""

    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported README language: {language}")
    header = (
        "| Scope | Status | Evidence available today |\n|---|---|---|"
        if language == "en"
        else "| 范围 | 状态 | 当前证据 |\n|---|---|---|"
    )
    rows = [header]
    for reference, entry in iter_matrix_entries(status):
        display = _localized(entry, "display", language, reference)
        evidence = _localized(entry, "evidence_summary", language, reference)
        claim = str(entry.get("claim", ""))
        try:
            claim_label = CLAIM_LABELS[claim][language]
        except KeyError as exc:
            raise ValueError(f"{reference} has no README label for claim {claim!r}") from exc
        rows.append(f"| {display} | **{claim_label}** | {evidence} |")
    return f"{START_MARKER}\n" + "\n".join(rows) + f"\n{END_MARKER}"


def replace_readme_matrix(text: str, status: dict[str, Any], language: str) -> str:
    """Replace exactly one managed product-status block."""

    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one product-status marker pair")
    start = text.index(START_MARKER)
    end = text.index(END_MARKER, start) + len(END_MARKER)
    return text[:start] + render_readme_matrix(status, language) + text[end:]


def extract_readme_matrix(text: str) -> str:
    """Extract the managed matrix including its markers."""

    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one product-status marker pair")
    start = text.index(START_MARKER)
    end = text.index(END_MARKER, start) + len(END_MARKER)
    return text[start:end]


def _localized(
    entry: dict[str, Any],
    field: str,
    language: str,
    reference: str,
) -> str:
    value = entry.get(field)
    if not isinstance(value, dict) or not str(value.get(language, "")).strip():
        raise ValueError(f"{reference}.{field}.{language} is required")
    return str(value[language]).replace("|", "\\|").strip()


__all__ = [
    "CLAIM_LABELS",
    "END_MARKER",
    "START_MARKER",
    "extract_readme_matrix",
    "render_readme_matrix",
    "replace_readme_matrix",
]
