"""Offline adapter for original G5 snapshots; never calls a model or World."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from rosclaw.connectors.ros.engineering import compile_engineering_case, summarize_case
from rosclaw.connectors.ros.intelligence import RosSystemModel


def replay(study, output):
    study = Path(study).resolve()
    output = Path(output).resolve()
    if output == study or study in output.parents:
        raise ValueError("output must be outside originals")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(study.glob("episode-*/original-result.json")):
        row = json.loads(path.read_text())
        worlds = [
            Path(w["directory"])
            for w in row["original_operator_Worlds"]
            if w["case"] != "positive" and w["returncode"] == 0
        ]
        world = worlds[-1]
        e = world / "execution"
        system = e / "engineering-original-system.json"
        model = RosSystemModel.from_dict(json.loads(system.read_text()))
        urdf = (world / "assets/robot.urdf").read_bytes()
        requirements = json.loads((e / "engineering-original-requirements.json").read_text())
        now = datetime.fromisoformat(
            json.loads((e / "engineering-original-candidate-review.json").read_text())[
                "reviewed_at"
            ]
        )
        options = {"source_ref": str(system), "now": now, "evidence_domain": "SEALED_REPLAY"}
        case = compile_engineering_case(model, urdf, requirements, **options)
        again = compile_engineering_case(model, urdf, requirements, **options)
        assert case == again
        summary = summarize_case(case)
        characters = len(json.dumps(summary, ensure_ascii=False))
        assert characters <= 2500
        (output / f"episode-{row['ordinal']:02d}-case.json").write_text(
            case.model_dump_json(indent=2) + "\n"
        )
        (output / f"episode-{row['ordinal']:02d}-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        rows.append(
            {
                "ordinal": row["ordinal"],
                "original_case_label_for_review_only": row["case"],
                "case_hash": case.case_hash,
                "summary_characters": characters,
                "statuses": {x.id: x.status for x in case.constraints},
                "candidates": [x.semantic_field for x in case.repair_candidates],
            }
        )
    if len(rows) != 54:
        raise ValueError("54 original snapshots required")
    result = {
        "status": "PASS_54_DETERMINISTIC_READ_ONLY_REPLAYS_NOT_MODEL_OR_PHYSICAL_ACCEPTANCE",
        "rows": rows,
        "models_invoked": 0,
        "Worlds_started": 0,
        "authority": "NONE",
    }
    (output / "replay-review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("output")
    a = p.parse_args()
    result = replay(a.study, a.output)
    print(result["status"])


if __name__ == "__main__":
    main()
