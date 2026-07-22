# ROSClaw × LeRobot Bridge v1.0.1 — Compatibility Matrix

Validated 2026-07-17 on the Jetson (aarch64) + Inspire RH56 left hand;
Agent discoverability + developer black-box closed 2026-07-22 on x86_64
(no physical hand) via MCP action tools.

```yaml
bridge_version: "1.0.1"
release_tag: rosclaw-lerobot-bridge-v1.0.1
branch: main
validated_commit: TBD-at-tag-time   # final main SHA of the v1.0.1 closure PR

rosclaw:
  core_python: "3.12"          # 3.12.3 (also runs 3.11 per CI matrix)

lerobot:
  tested_version: "0.6.0"      # pip: lerobot>=0.6,<0.7
  worker_python: "3.12"        # isolated runtime ~/.rosclaw/envs/lerobot
  torch: "2.11.0+cu130"        # aarch64 wheel from PyPI, CUDA available

policy_runtime:
  active_sessions: 1
  dtype: auto
  execution_mode:
    - proposal_only
    - shadow                   # 1000 steps @5Hz OBSERVED gate pass
    - single_step_receding_horizon   # REAL via ActionGateway rh56.single_step

agent_surface:
  mcp_tools: 22                # 7 core + 6 body + 4 control-plane + 5 product
  product_status_tool: get_product_status
  action_entry: request_action # SHADOW/FIXTURE always; REAL requires operator permit
  receipt_tools: [get_action_status, get_execution_receipt, explain_execution]
  guidance: AGENTS.md / CLAUDE.md / ROSCLAW.md / .agents skill (managed blocks)
  agent_blackbox: developer_agent_blackbox (independent: false)

reference_policy:
  type: rosclaw_rh56_reference
  action_dim: 6
  plugin: lerobot-policy-rosclaw-rh56 0.1.0

reference_body:
  model: inspire_rh56_left (this rig; right hand profile identical except
    device/slave)
  transport: rs485_modbus_rtu (FTDI FT232R, 115200 8N1, slave id 1)
  position_range: [0, 1000]
  feedback:
    status_registers: 3        # this firmware groups STATUS/TEMP by pair
    temperature_registers: 3
  calibration:
    tolerances_raw: {fingers: 12, thumb: 40, thumb_rot: 40}
    force_soft_limit_g: 100
    force_hard_limit_g: 300
    temperature_warning_c: 55
    temperature_stop_c: 60
    thresholds_source: measured_conservative

evidence:
  exp0_modbus_audit: /tmp/exp0_modbus_audit.json
  exp1_calibration: configs/rh56_left_01_calibration.yaml (validated, mock=false)
  exp2_shadow_gate: reports/lerobot_bridge/P5_RH56_SHADOW_REPORT_S2_OBSERVED.md
  exp3_graded_execution: reports/lerobot_bridge/P5D_GRADED_REAL_EXECUTION_REPORT.md
  exp4_fault_injection: reports/lerobot_bridge/P5D_REAL_FAULT_INJECTION_REPORT.md
  practice_root: ~/.rosclaw/practice/runs/lerobot_bridge (verify --strict pass)

not_supported:
  - CAN RH56 execution           # MCPExecutor CAN path stays fail-closed stub
  - open_loop_action_chunks      # single-step receding horizon only
  - multiple_active_sessions
  - unattended_execution         # operator-armed permits + physical estop required
  - usb_unplug_hot_swap_during_motion  # fault injection done in STATIC pose only
```

## Notes

- The mock `P5_RH56_REAL_SHADOW_REPORT.md` (fixture evidence with a misleading
  "REAL" name) is superseded by the OBSERVED report above; the mock report
  never lived in this repo, so nothing needed renaming.
- Deadline budget: serial read ≈75 ms + inference ≈1 ms + mapping/sandbox
  <0.1 ms per step → 5 Hz shadow with 0 deadline misses.
- Position verification tolerances are per-actuator measured values
  (steady-state error + in-motion lag), not vendor specs.
