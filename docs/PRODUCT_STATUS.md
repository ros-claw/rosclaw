# Product Status Source

`src/rosclaw/product/status.yaml` is the canonical public capability status
for ROSClaw. It records release maturity, golden-path execution modes, support
tiers, Agent black-box readiness, and evidence references.

## Claim Rules

- `verified` requires a non-fixture Evidence ID.
- Simulation `verified` requires non-fixture evidence with
  `observation_scope: physics_simulation`.
- Hardware `verified` requires independent, non-fixture evidence with
  `observation_scope: physical_hardware`.
- Agent black-box `verified` requires independent, non-fixture evidence with
  `observation_scope: external_agent_simulation` or
  `external_agent_hardware`.
- `developer_observed` is not rendered as verified.
- Fixture evidence cannot support a real-hardware claim.
- `agent_ready: true` requires a verified Agent black-box dimension.
- H2 through H5 support tiers require the corresponding verified dimension and
  its scoped evidence.
- The release version must match `rosclaw.__version__`.

RH56 developer-run hardware reports remain `developer_observed` until the
independent v1 hardware revalidation and Agent black-box acceptance finish.

## Synchronization

Update the YAML, then regenerate the managed README matrices:

```bash
python scripts/product/render_status.py
python scripts/product/render_status.py --check
```

`rosclaw status capabilities --json` exposes the same source for scripts and
other product surfaces.
