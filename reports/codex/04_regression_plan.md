# Regression Plan

## P1: SeekDB Operations

- Add a deployment integration job that starts `oceanbase/seekdb` and runs the real MySQL DSN Practice loop.
- Decide whether RuntimeConfig should add a `mysql`/`seekdb` backend alongside `memory` and `sqlite`.
- Deprecate or clearly package the legacy HTTP `SeekDBBridge` adapter path.

## P2: Provider Reality

- Re-run the official DeepSeek invocation after the account has balance.
- Run latency/schema benchmarks against configured VLM/VLA/world-model endpoints.
- Record endpoint health and benchmark evidence in Practice rather than treating the built-in catalog as runtime health.

## P2: Hub Authentication

- Exercise authenticated publish/verify/install when a non-production Hub test
  token is available.
- Keep public Hub dry-run as a zero-write regression gate.

## P2: Evolution Hardening

- Keep `tests/integration/test_physical_ai_agent_acceptance.py` as the
  no-hardware acceptance chain for Runtime -> Practice/Memory -> How -> Auto
  -> sandbox/Darwin -> simulated Skill Registry promotion.
- Add production benchmark datasets and explicit human approval before any
  promotion beyond the simulated `sim` level.

## P3: Hardware Acceptance

- Validate one explicitly authorized robot through read-only state, simulation preview, guarded command validation, and supervised execution.
- Keep certified emergency-stop and industrial safety systems outside ROSClaw's software trust boundary.
