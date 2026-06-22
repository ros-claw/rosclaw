# ROSClaw CLI

This reference documents the `rosclaw` command-line interface.

## Status Labels

| Label | Meaning |
|---|---|
| **Stable** | Implemented and tested. Safe for primary workflows. |
| **Experimental** | Implemented but API may change. Use with caution. |
| **Planned** | Documented design, not yet implemented. |
| **Research** | Research-facing workflow; may not ship in v1.x. |

Use `rosclaw --help` and `rosclaw <command> --help` to see the current help text.

---

## Core

| Command | Status | Description |
|---|---|---|
| `rosclaw firstboot` | Stable | Initialize local Physical-AI runtime workspace |
| `rosclaw doctor` | Stable | Run health diagnosis |
| `rosclaw --version` | Stable | Show version |
| `rosclaw init [DIR]` | Stable | Initialize a ROSClaw workspace |
| `rosclaw run` / `rosclaw start` | Stable | Start the ROSClaw runtime |
| `rosclaw stop` | Stable | Stop the ROSClaw runtime |
| `rosclaw restart` | Stable | Restart the ROSClaw runtime |
| `rosclaw status` | Stable | Show runtime status |
| `rosclaw logs` | Stable | Show runtime logs |
| `rosclaw uninstall` | Stable | Uninstall ROSClaw |

---

## Agent

| Command | Status | Description |
|---|---|---|
| `rosclaw agent init claude-code` | Stable | Generate MCP and agent context files |
| `rosclaw agent doctor claude-code` | Stable | Diagnose Claude Code integration |
| `rosclaw agent test claude-code` | Stable | Validate generated agent files |

---

## Body

| Command | Status | Description |
|---|---|---|
| `rosclaw body init --robot unitree-g1` | Stable | Initialize a new body instance from an e-URDF profile |
| `rosclaw body create --robot unitree-g1 --name g1_01` | Stable | Create a body instance |
| `rosclaw body link-eurdf unitree-g1` | Stable | Link current body to an e-URDF profile |
| `rosclaw body inspect` | Stable | Inspect current body state |
| `rosclaw body show` | Stable | Show body summary |
| `rosclaw body state` | Stable | Print unified body state |
| `rosclaw body query "question"` | Stable | Ask a question about the body |
| `rosclaw body validate` | Stable | Validate body workspace |
| `rosclaw body render` | Stable | Re-render EMBODIMENT.md and summaries |
| `rosclaw body switch <body_id>` | Stable | Switch active body |
| `rosclaw body remove <body_id>` | Stable | Remove a body instance |
| `rosclaw body history` | Stable | List body snapshots |
| `rosclaw body diff` | Stable | Compare body states |
| `rosclaw body update-state` | Stable | Update body instance state |
| `rosclaw body note "message"` | Stable | Add a maintenance/incident note |
| `rosclaw body calibration update --file calib.yaml` | Stable | Update calibration |
| `rosclaw body fault add --component X --severity high --summary "..."` | Stable | Add a known fault |
| `rosclaw body maintenance add --component X --summary "..."` | Stable | Add a maintenance event |
| `rosclaw body retrofit add` | Stable | Record a hardware retrofit |
| `rosclaw body capability enable/disable/degrade <cap_id>` | Stable | Manage capabilities |
| `rosclaw body export dest.zip` | Stable | Export body directory as archive |

---

## Provider

| Command | Status | Description |
|---|---|---|
| `rosclaw provider list` | Stable | List registered providers |
| `rosclaw provider invoke <provider_id> <input>` | Stable | Invoke a provider capability |
| `rosclaw provider init` | Planned | Initialize a provider configuration |
| `rosclaw provider route --capability vision_language_action` | Planned | Route a capability request |

---

## Sandbox

| Command | Status | Description |
|---|---|---|
| `rosclaw sandbox run --robot sim_ur5e --world tabletop --task reach` | Stable | Run a sandbox episode |
| `rosclaw sandbox validate <robot_id>` | Stable | Validate a robot in sandbox |
| `rosclaw sandbox check --robot <robot> --action <action>` | Stable | Check action safety |
| `rosclaw sandbox replay <episode_id>` | Stable | Replay a sandbox episode |
| `rosclaw sandbox list-worlds` | Stable | List available sandbox worlds |

---

## Practice

| Command | Status | Description |
|---|---|---|
| `rosclaw practice list` | Stable | List recorded episodes |
| `rosclaw practice show <episode_id>` | Stable | Show episode details |
| `rosclaw practice replay <episode_id>` | Stable | Replay episode trace |
| `rosclaw practice export <episode_id> --format json` | Stable | Export episode metadata |
| `rosclaw practice start --sources ...` | Planned | Start continuous praxis capture |

---

## Memory

| Command | Status | Description |
|---|---|---|
| `rosclaw memory query "last failed grasp near red cup"` | Stable | Query memory for similar experiences |
| `rosclaw memory explain` | Stable | Explain the most recent failure |
| `rosclaw memory status` | Stable | Show memory module status |

---

## How

| Command | Status | Description |
|---|---|---|
| `rosclaw how explain <episode_id>` | Stable | Explain a failure episode |
| `rosclaw how recover <episode_id>` | Stable | Generate a recovery plan |
| `rosclaw how advise --task ... --failure ...` | Planned | Ask for runtime repair advice |

---

## Know

| Command | Status | Description |
|---|---|---|
| `rosclaw know compile --task g1_kick_ball` | Stable | Compile a task into a TaskCard |
| `rosclaw know search "symptom"` | Stable | Search knowledge base |
| `rosclaw know robot <robot_id>` | Stable | Show robot safety limits and simulation profile |
| `rosclaw know recommend --task "pick and place"` | Stable | Recommend robots for a task |
| `rosclaw know validate --taskcard file.yaml` | Stable | Validate a TaskCard YAML file |
| `rosclaw know eval-taskcard --taskcard gen.yaml --gold gold.yaml` | Stable | Evaluate a TaskCard against a gold fixture |
| `rosclaw know export-hooks --taskcard file.yaml --out ./hooks` | Stable | Export hooks from a TaskCard |

---

## Auto

| Command | Status | Description |
|---|---|---|
| `rosclaw auto init --task pick_cube --skill reach` | Stable | Initialize an auto task |
| `rosclaw auto run --task pick_cube --rounds 50` | Stable | Run auto evolution |
| `rosclaw auto status --task pick_cube` | Stable | Show auto status |
| `rosclaw auto champion --task pick_cube` | Stable | Show current champion |
| `rosclaw auto deadends --task pick_cube` | Stable | List dead ends |
| `rosclaw auto report --task pick_cube` | Stable | Generate evolution report |
| `rosclaw auto run --suite tabletop_grasp` | Planned | Run an evolution suite |

---

## Darwin

| Command | Status | Description |
|---|---|---|
| `rosclaw darwin eval --skill pick_cube` | Research | Evaluate skill candidates under benchmark pressure |

`rosclaw-darwin` currently operates as a subcomponent of the Auto evolution pipeline. A standalone `darwin` CLI is on the research roadmap.

---

## Skill

| Command | Status | Description |
|---|---|---|
| `rosclaw skill list` | Stable | List available skills |
| `rosclaw skill invoke <skill_id> <input>` | Stable | Invoke a skill |
| `rosclaw skill check <skill_id>` | Stable | Check skill compatibility against current body |
| `rosclaw skill champions list` | Stable | List current champions |
| `rosclaw skill lineage <skill_id>` | Stable | Show skill lineage |
| `rosclaw skill rollback <skill_id> --to v1.0.0` | Stable | Rollback skill to version |

---

## Hub

| Command | Status | Description |
|---|---|---|
| `rosclaw hub search g1` | Stable | Search the local catalog index |
| `rosclaw hub login --registry https://hub.rosclaw.io --token $TOKEN` | Stable | Authenticate with a Hub registry |
| `rosclaw hub whoami` | Stable | Show active Hub identity |
| `rosclaw hub logout` | Stable | Forget stored credentials |
| `rosclaw hub sync` | Stable | Sync the local catalog index |
| `rosclaw hub validate ./manifest.yaml` | Stable | Validate an asset manifest |
| `rosclaw hub verify ./asset_dir` | Stable | Verify asset integrity |
| `rosclaw hub policy check ./asset_dir` | Stable | Check asset against local policy |
| `rosclaw hub ref parse rosclaw://...` | Stable | Parse a rosclaw URI |
| `rosclaw hub schema export` | Stable | Export manifest JSON Schema |
| `rosclaw hub install <asset>` | Planned | Install an asset |
| `rosclaw hub list --installed` | Planned | List installed assets |

---

## Dashboard

| Command | Status | Description |
|---|---|---|
| `rosclaw dashboard --open` | Stable | Start dashboard web server |
| `rosclaw dashboard` | Stable | Show dashboard status summary |

---

## Forge

| Command | Status | Description |
|---|---|---|
| `rosclaw forge sdk-to-mcp --name unitree_go2 --output ./out` | Stable | Convert SDK doc to MCP bundle |
| `rosclaw forge validate ./bundle` | Stable | Validate a Forge bundle |
| `rosclaw forge install ./bundle --staging` | Stable | Install a bundle to staging |

---

## Robot Registry

| Command | Status | Description |
|---|---|---|
| `rosclaw robot list` | Stable | List available robots |
| `rosclaw robot install <robot_id>` | Stable | Install/register a robot |
| `rosclaw robot inspect <robot_id>` | Stable | Show robot profile |
| `rosclaw robot validate <robot_id>` | Stable | Validate e-URDF completeness |

---

## Config / Profile

| Command | Status | Description |
|---|---|---|
| `rosclaw config show` | Stable | Show current config |
| `rosclaw config path` | Stable | Show config file path |
| `rosclaw config validate` | Stable | Validate config schema |
| `rosclaw config edit` | Stable | Open config in editor |
| `rosclaw profile list` | Stable | List profiles |
| `rosclaw profile current` | Stable | Show active profile |
| `rosclaw profile use offline` | Stable | Activate a profile |

---

## Runtime

| Command | Status | Description |
|---|---|---|
| `rosclaw runtime backends` | Stable | List available runtime backends |

---

## Firewall

| Command | Status | Description |
|---|---|---|
| `rosclaw firewall check --robot <robot> --action <action>` | Stable | Check action safety |

---

## Events

| Command | Status | Description |
|---|---|---|
| `rosclaw events tail` | Stable | Tail EventBus events |
| `rosclaw events publish <topic>` | Stable | Publish an event |
| `rosclaw events list` | Stable | List published events |

---

## MCP

| Command | Status | Description |
|---|---|---|
| `rosclaw mcp serve --transport stdio` | Stable | Start the P0 MCP server |
| `rosclaw mcp install` | Stable | Install a hardware MCP server |
| `rosclaw mcp list` | Stable | List hardware MCP servers |
| `rosclaw mcp health` | Stable | Check hardware MCP health |

---

## Sense

| Command | Status | Description |
|---|---|---|
| `rosclaw sense now` | Stable | Get current BodySense snapshot |
| `rosclaw sense state` | Stable | Show detailed BodyState |
| `rosclaw sense readiness --task "walk"` | Stable | Show body readiness |
| `rosclaw sense watch` | Stable | Watch body sense stream |
| `rosclaw sense events` | Stable | Show recent body events |
| `rosclaw sense explain --task "walk"` | Stable | Explain task block |

---

## Demo

| Command | Status | Description |
|---|---|---|
| `rosclaw demo tabletop-grasp --robot-id ur5e` | Experimental | Tabletop grasp demo |
| `rosclaw demo mobile-pid --robot-id turtlebot` | Experimental | Mobile base PID control demo |

---

## ROS Connector

| Command | Status | Description |
|---|---|---|
| `rosclaw ros discover` | Stable | Discover ROS capabilities |
| `rosclaw ros ping` | Stable | Ping ROS bridge |
| `rosclaw ros list-capabilities` | Stable | List ROS capabilities |
| `rosclaw ros emergency-stop` | Stable | Emergency stop |

---

## Notes

- Commands marked **Planned** are documented for roadmap transparency but do not yet have implementations. Use `rosclaw --help` to confirm what is available in your installed version.
- Commands marked **Research** are part of active research and may change significantly or be removed.
