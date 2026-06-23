"""Evidence artifact helpers for skill lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw.skill.models import EvalReport, MiningReport


def write_eval_report(root: Path, report: EvalReport) -> Path:
    candidate = report.candidate_id or "default"
    path = root / "evidence" / "reports" / f"{candidate}_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_mining_report(root: Path, report: MiningReport) -> Path:
    path = root / "evidence" / "reports" / f"{report.candidate_id}_mining.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def latest_eval_report(root: Path, candidate_id: str | None = None) -> Path | None:
    reports_dir = root / "evidence" / "reports"
    if not reports_dir.exists():
        return None
    if candidate_id:
        path = reports_dir / f"{candidate_id}_eval.json"
        return path if path.exists() else None
    # Return most recent eval report.
    candidates = sorted(reports_dir.glob("*_eval.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_eval_report_dict(root: Path, candidate_id: str | None = None) -> dict[str, Any] | None:
    path = latest_eval_report(root, candidate_id)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))
