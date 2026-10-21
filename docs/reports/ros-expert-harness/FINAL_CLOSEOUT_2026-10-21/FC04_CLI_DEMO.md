# Reproducible existing-CLI demonstration

This is a historical offline replay with preserved Source/lifecycle unknowns,
not an eligible P1, a new model trial, current L1, or a new physics experiment.
Inputs are exact copies with SHA256 identities; do not replace their original
review clock with today's time. Run from a checkout with its existing .venv:

```bash
FC_DEMO_DIR=$(mktemp -d)
ROSCLAW_HOME="$FC_DEMO_DIR/home" .venv/bin/python -m rosclaw.entrypoint ros engineering-case \
  --snapshot docs/reports/ros-expert-harness/FINAL_CLOSEOUT_2026-10-21/demo-inputs/original-system.json \
  --urdf docs/reports/ros-expert-harness/FINAL_CLOSEOUT_2026-10-21/demo-inputs/original-robot.urdf \
  --requirements docs/reports/ros-expert-harness/FINAL_CLOSEOUT_2026-10-21/demo-inputs/original-requirements.json \
  --reviewed-at '2026-10-15T05:23:02.596727+00:00' \
  --plan-staging \
  --staging-nav2 docs/reports/ros-expert-harness/FINAL_CLOSEOUT_2026-10-21/demo-inputs/original-nav2.yaml \
  --staging-bringup docs/reports/ros-expert-harness/FINAL_CLOSEOUT_2026-10-21/demo-inputs/original-bringup.json \
  --output "$FC_DEMO_DIR/case.json" --json
```

Expected: `ok=true`, authority NONE, L1 NOT_EVALUATED; a stage-only conditional
TF diff with rollback and required new-World proofs. F-SOURCE-001 and F-LIFE-001
remain UNKNOWN. This demonstrates why a valid static proposal alone cannot pass
the final P1 entry. It starts no World and applies no configuration.

To inspect this final protocol without running a model:

```bash
PYTHONPATH=src:. .venv/bin/python -c "from pathlib import Path; from benchmarks.ros_expert_ablation.closeout_protocol import CloseoutProtocolV1; p=CloseoutProtocolV1.model_validate_json(Path('docs/reports/ros-expert-harness/FINAL_CLOSEOUT_2026-10-21/FC01_CLOSEOUT_PROTOCOL.json').read_bytes()); print(p.original_entry_gate())"
```

Expected: `NO_GO_ORIGINAL_CASE_MATRIX_INCOMPLETE`. Do not change UNAVAILABLE
to ORIGINAL_CAPTURE to bypass the audit. New evidence would belong to a separate
future study, not a retry of this frozen closeout.
