# Non-Hardware Closure

Date: 2026-07-09

Evidence directory: `reports/codex/20260709_074712`

## Closed

- ROS1 published-port mismatch: compose forwards port 9091 through
  `websocket_external_port`; ping, Noetic discovery, manifest compilation, and
  read-only pose subscription pass.
- Isolated type checking: `.venv-codex/bin/mypy src/rosclaw` passes 457 files.
- Provider CLI: DeepSeek is registered directly, custom providers are invoked
  without an incorrect reasoner fallback, upstream provider errors return
  non-zero, and HTTP errors include the upstream message.
- Provider protocol: a real local HTTP server returns a successful DeepSeek
  compatible response in the regression suite.
- Public MCP Hub: `ros-claw/g1-mcp` resolves to
  `io.rosclaw.hub.ros-claw.g1-mcp@0.1.0`; dry-run writes zero files.
- Full runtime: 3712 tests pass; MuJoCo, agent MCP, Practice, exports, and real
  SeekDB all pass with `FAILURES=0`.

## External Conditions

- The supplied DeepSeek credential reaches `https://api.deepseek.com` but the
  service returns `402 Insufficient Balance`. This is now reported clearly and
  cannot be changed in repository code.
- Authenticated Hub publishing was not attempted because no Hub write token was
  supplied. Public read-only registry access is verified.
- Real hardware was intentionally not exercised.
