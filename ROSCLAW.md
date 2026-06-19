# ROSCLAW.md — Physical AI Runtime Manifest

This file is the authoritative boundary description for the ROSClaw runtime in
this project. It is read by Claude Code on every session start. Human edits
outside managed blocks are preserved by `rosclaw agent init claude-code`.

<!-- ROSCLAW-MANAGED-BEGIN -->
## Runtime profile (managed)

- **Project root:** `/home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0`
- **MCP transport:** `stdio`
- **Robot ID:** (none detected)

## Validate-before-motion workflow

1. Agent proposes motion via `validate_trajectory`.
2. `validate_trajectory` returns `{"is_safe": true}` **only** when the plan
   passes the firewall gate and sandbox simulation.
3. If safe, the agent **must** ask a human operator before sending any real
   command to ROS/hardware.
4. `sandbox_run` may be used to preview physics in MuJoCo; it never commands
   real hardware.
5. On unexpected behavior, call `emergency_stop` and follow local E-stop
   procedures.

## Deny rules

Claude Code must never run these commands directly:

- `rostopic pub /cmd_vel ...`
- `ros2 topic pub /cmd_vel ...`
- Any direct motor/DDS/hardware write without operator confirmation
- Any `sudo` command on the robot host without explicit justification
<!-- ROSCLAW-MANAGED-END -->

## Maintainer notes

_Add operational notes here. They will be preserved across init runs._
