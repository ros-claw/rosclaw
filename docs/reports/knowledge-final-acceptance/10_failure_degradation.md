# Failure and degradation

Status: PASS for fail-closed behavior. External unavailability is explicit.

| Failure | Behavior |
|---|---|
| Know unavailable | How abstains with reason |
| Empty Reference Pack | How abstains; no uncited recommendation |
| All candidates incompatible | How abstains and preserves rejection records |
| Unresolved source contradiction | Know rejects candidate use and explain reports reason |
| Rerank model unavailable | Capability false; deterministic RRF fallback |
| MCP timeout/rate error | Adapter warning/degraded research result; other sources continue |
| Whole-run deadline or token/byte/doc limit | Collection stops with a named budget warning |
| Stale cached pack | Marked cached+stale; never relabelled fresh |
| Invalid upstream wire payload | Rejected, not hidden by cache |

The vLLM paired acceptance is not complete. The endpoint first queued beyond
bounded retries and later refused TCP connection. No model-quality result was
fabricated.

The broad Core test run reached 3,579 passed before being stopped after eight
environment/pre-existing failures and four Docker errors. Failures involved a
present Codex binary invalidating “missing binary” fixtures, CLI PATH setup,
unconfigured LeRobot, a legacy Core schema incompatibility with the temporary
SeekDB server, and Docker Hub timeout for a ROS image. The knowledge subset
(64/64) and Node agent suite (32/32) passed. These unrelated failures are not
represented as a green full-Core result.
