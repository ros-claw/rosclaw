# Module Audit

Date: 2026-07-09

| Area | Verified implementation | Current boundary |
|---|---|---|
| Runtime | Runtime wiring, idempotent start/stop, handlers, health, and full regression suite pass. | No real robot actuation was authorized or attempted. |
| EventBus | RuntimeBus to PracticeRecorder produces strict event envelopes and nine-event fixture traces. | Cross-process durability still depends on configured runtime transport. |
| Body / e-URDF | Body CLI and existing compatibility/render tests pass. | Every third-party robot profile still needs asset-specific validation. |
| Provider | Contract routing, dry-run benchmark, direct DeepSeek registration, real HTTP invocation, timeout, and structured upstream failure handling pass. | The supplied official account returns `402 Insufficient Balance`; model quality/latency needs a funded account. |
| Sandbox / Firewall | UR5e MuJoCo model loads and advances eight steps with non-empty qpos/qvel. | Hardware safety certification is outside ROSClaw's scope. |
| Practice | Fixture record, strict verify, distill, local/real SeekDB, five query modes, exports, and terminal safety artifact integrity pass. | Live recording quality depends on source adapters and clocks. |
| Memory | Memory, SQLite, in-memory, and real SeekDB server clients pass focused tests. | RuntimeConfig still selects memory/SQLite; direct server selection is currently exposed through Practice CLI/client APIs. |
| Know / How | Evidence from the RH56 fixture resolves into body cognition and intervention records. | Semantic quality beyond deterministic evidence needs domain datasets. |
| Auto / Darwin | Canonical Runtime/How events reach Auto; a safety failure produces a proposal and passes sandbox plus three-seed Darwin evaluation. | External benchmark suites and human promotion gates remain deployment responsibilities. |
| Skill | The acceptance loop registers only a simulated `sim` champion after the evaluation gate; package lineage/promotion/rollback tests pass. | Real skill execution requires body compatibility and explicit safety approval. |
| Hub / MCP | Full tests pass; universal agent MCP stdio probe discovers 13 tools; public Hub `ros-claw/g1-mcp` resolves to a zero-write dry-run plan. | Authenticated publish needs a Hub write token and remains unexercised. |
| Agent Integration | Temp-project install generates MCP config, guidance, skill, Claude settings, and context snapshot. | It is cross-agent onboarding, not a native plugin package for each framework. |
| ROS Connector | Read-only ping/discover/manifest/pose tests pass on ROS2 ports 9090/32887 and ROS1 Noetic port 9091; `/turtle1/cmd_vel` is high risk. | No command topic was published. |
| CLI | Root help and all task-required module help commands pass. | Commands that can actuate hardware still require explicit user authorization. |

## Safety Boundary

No ROS publish, motor command, serial write, DDS command, or real hardware action was executed. ROS validation was read-only; simulation used MuJoCo only.
