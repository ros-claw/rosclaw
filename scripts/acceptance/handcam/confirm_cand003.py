#!/usr/bin/env python3
"""PR-PE-10: cand_003 confirmation campaign runner (Physical Evolution Lab §9.5).

Frozen protocol (declared BEFORE any session; never adjusted by results):

* Candidate: cand_evo_rps_2026_01_003_b8a47dd5 (cooldown_every_n_rounds=5)
* Arms: A (no memory) / C (candidate) — paired per block with the SAME
  gesture seed; arm order alternates (Latin-square by block index);
* Regime bins by START temperature (both hands max):
    cold        36–40 °C
    warm        43–46 °C
    hot_but_safe 47–49 °C
  Start temp delta between arms of one block ≤ 3 °C; blocks outside
  every bin are marked ENVIRONMENTALLY_INVALID (kept as safety evidence,
  never deleted);
* Blocks are grouped into time windows (one window = one continuous
  campaign sitting); ≥ 3 valid blocks across ≥ 2 windows required;
* Every block: 40 rounds/session, realsense, strict practice verify;
* Verdict: evaluate_confirmation (PR-PE-7) — VALIDATED_EFFECTIVE /
  INSUFFICIENT_EVIDENCE / REFUTED.

Runs in the REPO venv (the driver subprocesses into the workspace env
for hardware, like the canary path).  Thermal gate per session start
(≤49 °C absolute ceiling — hot_but_safe's upper edge); 52 °C hardware
protection remains the backstop.
"""

from __future__ import annotations

import json
import sys
import time

REPO_SRC = "/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src"
sys.path.insert(0, REPO_SRC)

from rosclaw.evolution.hardware.contracts import load_config  # noqa: E402
from rosclaw.evolution.hardware.evidence import EvidenceManifest  # noqa: E402
from rosclaw.evolution.hardware.namespace import ExperimentNamespace  # noqa: E402
from rosclaw.evolution.hardware.orchestrator import DEFAULT_CONFIG  # noqa: E402
from rosclaw.evolution.hardware.session_driver import Rh56RpsWorkspaceDriver  # noqa: E402
from rosclaw.evolution.hardware.thermal import default_temp_probe  # noqa: E402
from rosclaw.evolution.physical.promotion_v2 import RegimeBlock, evaluate_confirmation  # noqa: E402

CANDIDATE_ID = "cand_evo_rps_2026_01_003_b8a47dd5"
CANDIDATE_PARAMS = {"cooldown_every_n_rounds": 5}
ROUNDS = 40
ABSOLUTE_MAX_START_C = 49.0
MAX_ARM_TEMP_DELTA_C = 3.0
REGIME_BINS = {
    "cold": (36.0, 40.0),
    "warm": (43.0, 46.0),
    "hot_but_safe": (47.0, 49.0),
}


def _regime_bin(temp: float) -> str | None:
    for name, (lo, hi) in REGIME_BINS.items():
        if lo <= temp <= hi:
            return name
    return None


def _temps() -> float:
    probe = default_temp_probe()
    values = [v for v in probe.values() if isinstance(v, (int, float)) and v > 0]
    return max(values) if values else float("nan")


def run_block(
    driver: Rh56RpsWorkspaceDriver,
    namespace: ExperimentNamespace,
    window_id: str,
    block_index: int,
    seed: int,
) -> dict:
    """One A/C paired block (arm order alternates by block index)."""
    arms = [
        ("A_no_memory", "no_memory", None),
        ("C_candidate", "candidate_canary", CANDIDATE_PARAMS),
    ]
    if block_index % 2 == 1:
        arms.reverse()
    sessions: list[dict] = []
    first_arm_start: float | None = None
    for arm_index, (arm_name, group, params) in enumerate(arms):
        start_temp = _temps()
        # Inter-arm thermal pacing (physics: one session heats ~4-7 °C, so
        # the second arm ALWAYS starts hotter without pacing): wait —
        # recorded — until the second arm starts within the protocol's
        # arm-delta of the FIRST arm's start.  Unreachable within the
        # budget → the block is honestly incomplete (never silently
        # mismatched).
        if arm_index > 0 and first_arm_start is not None:
            waited = 0.0
            deadline = time.time() + 900.0
            while (
                start_temp > first_arm_start + MAX_ARM_TEMP_DELTA_C and time.time() < deadline
            ):
                time.sleep(20.0)
                waited += 20.0
                start_temp = _temps()
            if start_temp > first_arm_start + MAX_ARM_TEMP_DELTA_C:
                sessions.append(
                    {
                        "arm": arm_name,
                        "blocked": f"arm pacing unreachable after {waited:.0f}s "
                        f"(start {start_temp}°C > first {first_arm_start}°C + {MAX_ARM_TEMP_DELTA_C})",
                    }
                )
                continue
        if start_temp > ABSOLUTE_MAX_START_C:
            sessions.append(
                {"arm": arm_name, "blocked": f"start {start_temp}°C > {ABSOLUTE_MAX_START_C}°C"}
            )
            continue
        if first_arm_start is None:
            first_arm_start = start_temp
        out_dir = (
            namespace.evidence_root / "confirmation" / f"{window_id}_b{block_index}_{arm_name}"
        )
        if params is None:
            result = driver.run_session(
                group=group, seed=seed, rounds=ROUNDS, camera_source="realsense", out_dir=out_dir
            )
        else:
            result = driver.run_canary(
                candidate_id="confirm_cand003",
                candidate_params=params,
                seed=seed,
                rounds=ROUNDS,
                out_dir=out_dir,
            )
        sessions.append(
            {
                "arm": arm_name,
                "practice_id": result.practice_id,
                "invalid_rate": result.summary.get("invalid_rate"),
                "peak_temperature": result.summary.get("peak_temperature"),
                "start_temperature": start_temp,
                "seed": seed,
            }
        )
    done = [s for s in sessions if "invalid_rate" in s and s["invalid_rate"] is not None]
    if len(done) == 2:
        a = next(s for s in done if s["arm"] == "A_no_memory")
        c = next(s for s in done if s["arm"] == "C_candidate")
        delta = abs((a.get("start_temperature") or 0) - (c.get("start_temperature") or 0))
        # Regime bin from the arms' OWN start temperatures, NaN-safe
        # (a failed probe read never fabricates out_of_bins).
        starts = [
            s["start_temperature"]
            for s in done
            if isinstance(s.get("start_temperature"), (int, float))
        ]
        bin_name = _regime_bin(min(starts)) if starts else None
        environmentally_invalid = bin_name is None or delta > MAX_ARM_TEMP_DELTA_C
        return {
            "complete": True,
            "regime_bin": bin_name or "out_of_bins",
            "environmentally_invalid": environmentally_invalid,
            "invalid_reason": (
                "out_of_bins"
                if bin_name is None
                else f"arm temp delta {delta:.1f} > {MAX_ARM_TEMP_DELTA_C}"
            )
            if environmentally_invalid
            else None,
            "starts": starts,
            "a_invalid": a["invalid_rate"],
            "c_invalid": c["invalid_rate"],
            "sessions": sessions,
            "safety_events": sum(1 for s in done if (s.get("peak_temperature") or 0) >= 52),
        }
    return {"complete": False, "sessions": sessions, "safety_events": 0}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, help="time window id, e.g. 20260730T23")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=20263000)
    args = parser.parse_args()

    config = load_config(str(DEFAULT_CONFIG))
    namespace = ExperimentNamespace.from_config(config)
    manifest = EvidenceManifest.open(
        namespace.evidence_root, config.experiment_id, config.config_hash
    )
    driver = Rh56RpsWorkspaceDriver(config, namespace.practice_root)

    results = []
    for block_index in range(args.blocks):
        seed = args.seed_start + block_index
        block = run_block(driver, namespace, args.window, block_index, seed)
        block["window"] = args.window
        block["block_index"] = block_index
        results.append(block)
        manifest.record("confirmation_block", candidate_id=CANDIDATE_ID, **block)
        print(json.dumps(block, ensure_ascii=False, default=str)[:400], flush=True)

    # Evaluate with ALL confirmation blocks in the manifest (cross-window).
    entries = [e for e in manifest.by_kind("confirmation_block") if e.get("complete")]
    blocks = [
        RegimeBlock(
            regime_bin=str(e.get("regime_bin")),
            arm_a_invalid=float(e["a_invalid"]),
            arm_c_invalid=float(e["c_invalid"]),
            start_temp_delta_c=abs(
                (e["sessions"][0].get("start_temperature") or 0)
                - (e["sessions"][1].get("start_temperature") or 0)
            ),
            time_window=str(e.get("window")),
            environmentally_invalid=bool(e.get("environmentally_invalid")),
            safety_events=int(e.get("safety_events") or 0),
        )
        for e in entries
    ]
    report = evaluate_confirmation(blocks)
    manifest.record(
        "confirmation_evaluation",
        candidate_id=CANDIDATE_ID,
        report=report.to_dict(),
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
