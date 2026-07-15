# ROSClaw CLI Command Form

> Auto-generated from `src/rosclaw/cli.py` argparse definitions.
> Generated: 2026-06-23

这份表单根据当前 `src/rosclaw/cli.py` 中的 argparse 注册信息，列出所有 `rosclaw` 子命令、说明和用法。

## Core commands

| Command | Description | Usage |
|---------|-------------|-------|
| `rosclaw` | ROSClaw - Self-Evolving Runtime Infrastructure for Physical AI & Embodied Agents | `rosclaw [-h] [--version] {init,run,start,status,stop,restart,dashboard,doctor,firstboot,config,profile,uninstall,logs,events,ros,body,hub,robot,how,provider,auto,skill,sandbox,runtime,firewall,forge,memory,practice,know,sense,fleet,demo,agent,mcp} ...` |
| `init` | Initialize a ROSClaw workspace | `rosclaw init [-h] [--force] [dir]` |
| `run` | Start ROSClaw runtime | `rosclaw run [-h] [--robot-id ROBOT_ID] [--model-path MODEL_PATH] [--firewall] [--memory] [--practice] [--swarm]` |
| `start` | Start ROSClaw runtime | `rosclaw start [-h] [--robot-id ROBOT_ID] [--model-path MODEL_PATH] [--firewall] [--memory] [--practice] [--swarm]` |
| `status` | Show runtime status | `rosclaw status [-h] [--json]` |
| `stop` | Stop ROSClaw runtime | `rosclaw stop [-h]` |
| `restart` | Restart ROSClaw runtime | `rosclaw restart [-h] [--robot-id ROBOT_ID] [--model-path MODEL_PATH] [--firewall] [--memory] [--practice] [--swarm]` |
| `dashboard` | Open/show dashboard | `rosclaw dashboard [-h] [--open]` |
| `doctor` | Run health diagnosis | `rosclaw doctor [-h] [--ros2] [--ros] [--endpoint ENDPOINT] [--bootstrap] [--full] [--fix] [--json] [--gpu] [--network]` |
| `firstboot` | Run ROSClaw first boot wizard | `rosclaw firstboot [-h] [--yes] [--workspace WORKSPACE] [--profile {offline,cloud,hybrid}] [--robot ROBOT] [--safety {strict,moderate,relaxed}] [--enable-sandbox] [--disable-sandbox] [--enable-mcp] [--disable-mcp] [--enable-ros2] [--enable-memory] [--enable-practice] [--enable-auto] [--telemetry] [--no-telemetry] [--dev] [--force] [--dry-run] [--json]` |
| `config` | Configuration commands | `rosclaw config [-h] {show,path,validate,edit} ...` |
| `profile` | Profile management | `rosclaw profile [-h] {list,use,current} ...` |
| `uninstall` | Uninstall ROSClaw | `rosclaw uninstall [-h] [--keep-data] [--purge]` |
| `logs` | Show runtime logs | `rosclaw logs [-h] [--tail TAIL] [--level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--module MODULE] [--files FILES]` |
| `events` | EventBus commands | `rosclaw events [-h] [--tail TAIL] {tail,publish,list} ...` |
| `ros` | ROS bridge commands | `rosclaw ros [-h] {ping,discover,compile,list-capabilities,inspect-capability,validate-capability,execute-capability,emergency-stop} ...` |
| `body` | Body / embodiment commands | `rosclaw body [-h] {init,create,switch,remove,list,validate,render,show,state,query,fault,maintenance,calibration,retrofit,capability,link-eurdf,inspect,diff,update-state,note,history,export,fleet-compat} ...` |
| `hub` | ROSClaw Hub asset discovery, verification, and lifecycle | `rosclaw hub [-h] {validate,ref,schema,login,whoami,logout,sync,search,verify,policy,install,uninstall,update,list,publish} ...` |
| `robot` | Robot registry commands | `rosclaw robot [-h] {list,install,inspect,validate} ...` |
| `how` | How recovery commands | `rosclaw how [-h] {explain,recover} ...` |
| `provider` | Provider commands | `rosclaw provider [-h] {list,invoke} ...` |
| `auto` | Auto self-evolution commands | `rosclaw auto [-h] {init,run,status,champion,deadends,report} ...` |
| `skill` | Skill commands | `rosclaw skill [-h] {list,check,invoke,champions,lineage,rollback,init,validate,mine,eval,promote,package,verify-package,upload} ...` |
| `sandbox` | Sandbox commands | `rosclaw sandbox [-h] {list-worlds,validate,run,replay,check} ...` |
| `runtime` | Runtime backend commands | `rosclaw runtime [-h] {backends} ...` |
| `firewall` | Firewall safety checks | `rosclaw firewall [-h] {check} ...` |
| `forge` | Forge bundle commands | `rosclaw forge [-h] {sdk-to-mcp,validate,install} ...` |
| `memory` | Memory commands | `rosclaw memory [-h] {status,query,explain} ...` |
| `practice` | Practice episode commands | `rosclaw practice [-h] {list,show,replay,export} ...` |
| `know` | Knowledge base queries | `rosclaw know [-h] {search,robot,recommend} ...` |
| `sense` | Body sense and readiness commands | `rosclaw sense [-h] {now,state,readiness,watch,events,explain} ...` |
| `fleet` | Fleet-wide body operations | `rosclaw fleet [-h] {status,stop} ...` |
| `demo` | Run demonstration scenarios | `rosclaw demo [-h] {mobile-pid,tabletop-grasp} ...` |
| `agent` | Agent onboarding and diagnostics | `rosclaw agent [-h] {install,init,doctor,test} ...` |
| `mcp` | MCP server commands | `rosclaw mcp [-h] {serve,install,list,health} ...` |

## Body commands

| Command | Description | Usage |
|---------|-------------|-------|
| `body init` | Initialize a new body instance from an e-URDF profile | `rosclaw body init [-h] --robot ROBOT [--profile PROFILE] [--name NAME] [--workspace WORKSPACE] [--force] [--no-alias] [--render] [--validate] [--body BODY]` |
| `body create` | Create a new body instance from an e-URDF profile | `rosclaw body create [-h] --robot ROBOT --name NAME [--nickname NICKNAME] [--workspace WORKSPACE] [--force] [--no-alias]` |
| `body switch` | Switch the active body | `rosclaw body switch [-h] [--workspace WORKSPACE] body_id` |
| `body remove` | Remove a body instance | `rosclaw body remove [-h] [--workspace WORKSPACE] [--archive] body_id` |
| `body list` | List registered bodies | `rosclaw body list [-h] [--workspace WORKSPACE] [--json]` |
| `body validate` | Validate body workspace | `rosclaw body validate [-h] [--json] [--workspace WORKSPACE] [--body BODY]` |
| `body render` | Force re-render of EMBODIMENT.md and summaries | `rosclaw body render [-h] [--workspace WORKSPACE] [--body BODY]` |
| `body show` | Show body summary | `rosclaw body show [-h] [--agent] [--workspace WORKSPACE] [--body BODY]` |
| `body state` | Print unified body state | `rosclaw body state [-h] [--json] [--workspace WORKSPACE] [--body BODY]` |
| `body query` | Ask a question about the body | `rosclaw body query [-h] [--workspace WORKSPACE] [--json] [--body BODY] question` |
| `body fault` | Manage known faults | `rosclaw body fault [-h] {add,resolve} ...` |
| `body fault add` | Add a known fault | `rosclaw body fault add [-h] --component COMPONENT --severity {low,medium,high,critical} --summary SUMMARY [--workspace WORKSPACE] [--body BODY]` |
| `body fault resolve` | Resolve a known fault | `rosclaw body fault resolve [-h] [--summary SUMMARY] [--workspace WORKSPACE] [--body BODY] fault_id` |
| `body maintenance` | Add maintenance event | `rosclaw body maintenance [-h] {add} ...` |
| `body maintenance add` | Add a maintenance event | `rosclaw body maintenance add [-h] [--type {maintenance,repair,inspection,incident,safety}] --component COMPONENT --summary SUMMARY [--workspace WORKSPACE] [--body BODY]` |
| `body calibration` | Calibration commands | `rosclaw body calibration [-h] {update} ...` |
| `body calibration update` | Update calibration from a YAML file | `rosclaw body calibration update [-h] --file FILE [--workspace WORKSPACE] [--body BODY]` |
| `body retrofit` | Record a hardware retrofit | `rosclaw body retrofit [-h] {add} ...` |
| `body retrofit add` | Record a retrofit | `rosclaw body retrofit add [-h] --component COMPONENT --type {sensor_install,tool_install,actuator_swap,structural_mod,other} --summary SUMMARY [--workspace WORKSPACE] [--body BODY]` |
| `body capability` | Manage capabilities | `rosclaw body capability [-h] {disable,degrade,enable} ...` |
| `body capability disable` | Disable a capability | `rosclaw body capability disable [-h] --reason REASON [--workspace WORKSPACE] [--body BODY] capability_id` |
| `body capability degrade` | Degrade a capability | `rosclaw body capability degrade [-h] [--mode {slow,sim_only,restricted_workspace,human_supervised}] --reason REASON [--workspace WORKSPACE] [--body BODY] capability_id` |
| `body capability enable` | Enable a capability | `rosclaw body capability enable [-h] [--after-validation AFTER_VALIDATION] [--workspace WORKSPACE] [--body BODY] capability_id` |
| `body link-eurdf` | Link current body to an e-URDF profile | `rosclaw body link-eurdf [-h] [--version VERSION] [--instance-id INSTANCE_ID] [--nickname NICKNAME] [--workspace WORKSPACE] [--force] [--mode {copy,lock-only}] [--body BODY] profile_id` |
| `body inspect` | Inspect current body state | `rosclaw body inspect [-h] [--json] [--agent] [--source-trace] [--capabilities] [--components] [--skills] [--body BODY]` |
| `body diff` | Compare body states | `rosclaw body diff [-h] [--against AGAINST] [--format {text,json,patch}] [--only ONLY] [--body BODY]` |
| `body update-state` | Update body instance state | `rosclaw body update-state [-h] [--set SET] [--unset UNSET] [--enable-capability ENABLE_CAPABILITY] [--disable-capability DISABLE_CAPABILITY] [--component-status COMPONENT_STATUS] [--reason REASON] [--source SOURCE] [--dry-run] [--no-skill-check] [--from-ros] [--ros-endpoint ROS_ENDPOINT] [--body BODY]` |
| `body note` | Add a maintenance/incident note | `rosclaw body note [-h] [--type {note,maintenance,calibration,incident,repair,inspection,safety}] [--severity {info,warning,critical}] [--affects AFFECTS] [--tags TAGS] [--author AUTHOR] [--body BODY] message` |
| `body history` | List body snapshots | `rosclaw body history [-h] [--json] [--workspace WORKSPACE] [--body BODY]` |
| `body export` | Export body directory as an archive | `rosclaw body export [-h] [--format {zip,tar}] [--workspace WORKSPACE] [--body BODY] dest` |
| `body fleet-compat` | Aggregate skill compatibility across all bodies | `rosclaw body fleet-compat [-h] [--workspace WORKSPACE] [--json] [--body BODY]` |

## Sense commands

| Command | Description | Usage |
|---------|-------------|-------|
| `sense now` | Get current BodySense snapshot | `rosclaw sense now [-h] [--mock MOCK] [--robot-id ROBOT_ID] [--json]` |
| `sense state` | Show detailed BodyState | `rosclaw sense state [-h] [--mock MOCK] [--robot-id ROBOT_ID] [--json]` |
| `sense readiness` | Show body readiness | `rosclaw sense readiness [-h] --task TASK [--mock MOCK] [--robot-id ROBOT_ID] [--json]` |
| `sense watch` | Watch body sense stream | `rosclaw sense watch [-h] [--mock MOCK] [--robot-id ROBOT_ID] [--interval INTERVAL] [--limit LIMIT]` |
| `sense events` | Show recent body events | `rosclaw sense events [-h] [--mock MOCK] [--robot-id ROBOT_ID] [--limit LIMIT] [--json]` |
| `sense explain` | Explain task block | `rosclaw sense explain [-h] --task TASK [--mock MOCK] [--robot-id ROBOT_ID]` |

## Skill commands

| Command | Description | Usage |
|---------|-------------|-------|
| `skill list` | List available skills | `rosclaw skill list [-h]` |
| `skill check` | Check skill availability and body compatibility | `rosclaw skill check [-h] [--all] [--json] [skill_id]` |
| `skill invoke` | Invoke a skill | `rosclaw skill invoke [-h] [--trace-id TRACE_ID] skill_id input` |
| `skill champions` | Skill champion management | `rosclaw skill champions [-h] {list} ...` |
| `skill lineage` | Show skill lineage | `rosclaw skill lineage [-h] skill_id` |
| `skill rollback` | Rollback skill to version | `rosclaw skill rollback [-h] --to TO skill_id` |
| `skill init` | Initialize a new skill package from a template | `rosclaw skill init [-h] [--robot ROBOT] [--category CATEGORY] [--namespace NAMESPACE] [--template TEMPLATE] [--output OUTPUT] [--force] name` |
| `skill validate` | Validate a skill package against the ROSClaw Skill Hub schema | `rosclaw skill validate [-h] [--name NAME] [--workspace WORKSPACE] [--json] [skill_dir]` |
| `skill mine` | Mine a candidate skill from a source artifact | `rosclaw skill mine [-h] --from SOURCE --task TASK [--robot ROBOT] [--output OUTPUT] [--candidate CANDIDATE] [--json]` |
| `skill eval` | Evaluate a skill candidate against evaluation evidence | `rosclaw skill eval [-h] [--name NAME] [--candidate CANDIDATE] [--mode {replay,sandbox}] [--save-evidence] [--json] [skill_dir]` |
| `skill promote` | Promote a skill to a new lifecycle stage/version | `rosclaw skill promote [-h] --to-version TO_VERSION [--stage STAGE] [--require-eval-pass] [--workspace WORKSPACE] [--json] skill_ref` |
| `skill package` | Package a skill into a distributable archive | `rosclaw skill package [-h] [--name NAME] [--output OUTPUT] [--format {tar.gz}] [--include-evidence {none,summary,full}] [--workspace WORKSPACE] [--json] [skill_dir]` |
| `skill verify-package` | Verify a packaged skill archive | `rosclaw skill verify-package [-h] [--json] archive` |
| `skill upload` | Upload a skill package to the ROSClaw Hub | `rosclaw skill upload [-h] [--name NAME] [--visibility {public,private,org,unlisted}] [--hub-base-url HUB_BASE_URL] [--api-key-env API_KEY_ENV] [--dry-run] [--force] [--workspace WORKSPACE] [--json] [skill_dir]` |

## Robot commands

| Command | Description | Usage |
|---------|-------------|-------|
| `robot list` | List available robots | `rosclaw robot list [-h]` |
| `robot install` | Install/register a robot | `rosclaw robot install [-h] robot_id` |
| `robot inspect` | Inspect robot profile | `rosclaw robot inspect [-h] [--json] robot_id` |
| `robot validate` | Validate robot e-URDF | `rosclaw robot validate [-h] robot_id` |

## Sandbox commands

| Command | Description | Usage |
|---------|-------------|-------|
| `sandbox list-worlds` | List available sandbox worlds | `rosclaw sandbox list-worlds [-h]` |
| `sandbox verify` | Run deterministic MuJoCo sandbox verification | `rosclaw sandbox verify [-h] [--case {ur5e-joint-preview}] [--robot ROBOT] [--world WORLD] [--steps STEPS] [--json]` |
| `sandbox validate` | Validate robot in sandbox | `rosclaw sandbox validate [-h] robot_id` |
| `sandbox run` | Run a sandbox episode | `rosclaw sandbox run [-h] --robot ROBOT [--world WORLD] --task TASK [--backend BACKEND] [--trace-id TRACE_ID]` |
| `sandbox replay` | Replay a sandbox episode | `rosclaw sandbox replay [-h] episode_id` |
| `sandbox check` | Check action safety in sandbox | `rosclaw sandbox check [-h] --robot ROBOT --action ACTION [--trace-id TRACE_ID]` |

## Practice commands

| Command | Description | Usage |
|---------|-------------|-------|
| `practice list` | List recorded episodes | `rosclaw practice list [-h]` |
| `practice show` | Show episode details | `rosclaw practice show [-h] [--json] episode_id` |
| `practice replay` | Replay episode trace | `rosclaw practice replay [-h] episode_id` |
| `practice record` | Record a deterministic fixture | `rosclaw practice record --fixture FIXTURE --out DATA_ROOT [--json]` |
| `practice verify` | Verify event envelopes and artifacts | `rosclaw practice verify practice_id --data-root DATA_ROOT [--strict] [--json]` |
| `practice distill` | Distill closed-loop evidence | `rosclaw practice distill practice_id --data-root DATA_ROOT [--json]` |
| `practice ingest-seekdb` | Ingest into SQLite or real SeekDB | `rosclaw practice ingest-seekdb practice_id [--seekdb-path FILE | --seekdb-url MYSQL_DSN] [--json]` |
| `practice query` | Query distilled Practice data | `rosclaw practice query {episodes,failures,body-cognition,sim2real,candidates,interventions,explain-episode,explain-failure} ...` |
| `practice export` | Export Practice artifacts | `rosclaw practice export practice_id --format {parquet,lerobot} --output PATH` |

## Memory commands

| Command | Description | Usage |
|---------|-------------|-------|
| `memory status` | Show memory status | `rosclaw memory status [-h]` |
| `memory query` | Query memory | `rosclaw memory query [-h] [--limit LIMIT] query` |
| `memory explain` | Explain last failure | `rosclaw memory explain [-h] [--task-id TASK_ID]` |

## Know commands

| Command | Description | Usage |
|---------|-------------|-------|
| `know search` | Search knowledge base | `rosclaw know search [-h] [--robot-id ROBOT_ID] query` |
| `know robot` | Show robot knowledge | `rosclaw know robot [-h] [--task TASK] robot_id` |
| `know recommend` | Recommend robots for task | `rosclaw know recommend [-h] task` |

## How commands

| Command | Description | Usage |
|---------|-------------|-------|
| `how explain` | Explain failure episode | `rosclaw how explain [-h] episode_id` |
| `how recover` | Generate recovery plan | `rosclaw how recover [-h] [--output OUTPUT] episode_id` |

## Auto commands

| Command | Description | Usage |
|---------|-------------|-------|
| `auto init` | Initialize an auto task | `rosclaw auto init [-h] --task TASK [--robot ROBOT] --skill SKILL [--env ENV] [--type {skill_tuning,failure_repair}]` |
| `auto run` | Run auto evolution | `rosclaw auto run [-h] --task TASK [--rounds ROUNDS] [--episodes EPISODES] [--dry-run] [--policy POLICY]` |
| `auto status` | Show auto status | `rosclaw auto status [-h] [--task TASK]` |
| `auto champion` | Show current champion | `rosclaw auto champion [-h] --task TASK` |
| `auto deadends` | List dead ends | `rosclaw auto deadends [-h] [--task TASK]` |
| `auto report` | Generate evolution report | `rosclaw auto report [-h] --task TASK [--output OUTPUT] [--format {md,json}]` |

## Provider commands

| Command | Description | Usage |
|---------|-------------|-------|
| `provider list` | List registered providers | `rosclaw provider list [-h]` |
| `provider health` | Show provider health/capability contracts | `rosclaw provider health [provider_id] [--json]` |
| `provider route` | Explain capability route selection | `rosclaw provider route --capability CAPABILITY [--json]` |
| `provider benchmark` | Build a no-call benchmark plan | `rosclaw provider benchmark --dry-run [--json]` |
| `provider invoke` | Invoke a provider capability | `rosclaw provider invoke provider_id [input] [--capability CAPABILITY] [--json]` |
| `provider diagnose` | Diagnose interfaces against an active body | `rosclaw provider diagnose [--body BODY] [--workspace WORKSPACE] [--json]` |

## Events commands

| Command | Description | Usage |
|---------|-------------|-------|
| `events tail` | Tail EventBus events | `rosclaw events tail [-h] [--tail TAIL]` |
| `events publish` | Publish an event | `rosclaw events publish [-h] [--payload PAYLOAD] [--source SOURCE] [--trace-id TRACE_ID] topic` |
| `events list` | List published events | `rosclaw events list [-h] [--limit LIMIT]` |

## Firewall commands

| Command | Description | Usage |
|---------|-------------|-------|
| `firewall check` | Check action safety | `rosclaw firewall check [-h] --robot ROBOT --action ACTION [--world WORLD] [--trace-id TRACE_ID]` |

## Runtime commands

| Command | Description | Usage |
|---------|-------------|-------|
| `runtime backends` | List available runtime backends | `rosclaw runtime backends [-h]` |

## Forge commands

| Command | Description | Usage |
|---------|-------------|-------|
| `forge sdk-to-mcp` | Convert SDK doc to MCP bundle | `rosclaw forge sdk-to-mcp [-h] --name NAME [--sdk-docs SDK_DOCS] --output OUTPUT` |
| `forge validate` | Validate a bundle | `rosclaw forge validate [-h] bundle_path` |
| `forge install` | Install a bundle | `rosclaw forge install [-h] [--staging] bundle_path` |

## Hub commands

| Command | Description | Usage |
|---------|-------------|-------|
| `hub validate` | Validate an asset manifest.yaml | `rosclaw hub validate [-h] [--json] manifest` |
| `hub ref` | Asset reference utilities | `rosclaw hub ref [-h] {parse} ...` |
| `hub ref parse` | Parse a rosclaw:// URI | `rosclaw hub ref parse [-h] [--json] ref` |
| `hub schema` | Manifest schema utilities | `rosclaw hub schema [-h] {export} ...` |
| `hub schema export` | Export the manifest JSON Schema | `rosclaw hub schema export [-h] [--format {json,yaml}] [--output OUTPUT]` |
| `hub login` | Authenticate with a Hub registry | `rosclaw hub login [-h] --registry REGISTRY --token TOKEN [--insecure-local]` |
| `hub whoami` | Show the active Hub registry identity | `rosclaw hub whoami [-h]` |
| `hub logout` | Forget stored Hub credentials | `rosclaw hub logout [-h] [--registry REGISTRY]` |
| `hub sync` | Sync the local catalog index | `rosclaw hub sync [-h] [--registry REGISTRY] [--clear]` |
| `hub search` | Search the local catalog index | `rosclaw hub search [-h] [--type TYPE] [--namespace NAMESPACE] [--official] [--license LICENSE] [--robot ROBOT] [--compatible] [--limit LIMIT] [--json] [query]` |
| `hub verify` | Verify asset integrity (checksums, artifacts, signatures) | `rosclaw hub verify [-h] [--no-signature] [--json] asset_dir` |
| `hub policy` | Check asset against local permission and license policy | `rosclaw hub policy [-h] {check} ...` |
| `hub policy check` | Check permissions and license policy | `rosclaw hub policy check [-h] [--allow-real-robot] [--accept-license] [--json] asset_dir` |
| `hub install` | Install an asset from a local directory or registry reference | `rosclaw hub install [-h] [--dry-run] [--yes] [--accept-license] [--no-mcp-merge] [--skip-health] [--no-verify-signature] [--allow-real-robot] [--allow-safety-config-changes] [--allow-network-inbound] [--json] asset_dir` |
| `hub uninstall` | Uninstall an asset by reference | `rosclaw hub uninstall [-h] [--yes] [--json] ref` |
| `hub update` | Update an installed asset from a local directory | `rosclaw hub update [-h] [--dry-run] [--yes] [--accept-license] [--no-mcp-merge] [--skip-health] [--no-verify-signature] [--allow-real-robot] [--allow-safety-config-changes] [--allow-network-inbound] [--json] ref asset_dir` |
| `hub list` | List installed assets | `rosclaw hub list [-h] [--installed] [--json]` |
| `hub publish` | Prepare and publish a ROSClaw Hub asset | `rosclaw hub publish [-h] [--dry-run] [--private] [--public] [--sign] [--registry REGISTRY] [--output OUTPUT] [--json] asset_dir` |

## Mcp commands

| Command | Description | Usage |
|---------|-------------|-------|
| `mcp serve` | Start the P0 MCP server | `rosclaw mcp serve [-h] [--transport {stdio,http,sse}] [--host HOST] [--port PORT] [--robot-id ROBOT_ID] [--project-root PROJECT_ROOT] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]` |
| `mcp install` | Install/register a Hardware MCP server | `rosclaw mcp install [-h] [--version VERSION] [--dry-run] [--allow-dangerous] [--conflict {abort,rename,replace}] [--offline] [--skip-body] [--skip-claude] [--json] [--project-root PROJECT_ROOT] [--yes] alias` |
| `mcp list` | List installed and available Hardware MCP servers | `rosclaw mcp list [-h] [--installed] [--available] [--type TYPE_FILTER] [--bound] [--unbound] [--offline] [--json] [--project-root PROJECT_ROOT]` |
| `mcp health` | Run health checks for installed Hardware MCP servers | `rosclaw mcp health [-h] [--full] [--json] [--project-root PROJECT_ROOT] [server_name]` |

## Agent commands

| Command | Description | Usage |
|---------|-------------|-------|
| `agent install` | Install cross-agent ROSClaw onboarding files. | `rosclaw agent install [-h] [--project-root PROJECT_ROOT] [--profile PROFILE] [--robot ROBOT] [--transport {stdio,http,sse}] [--host HOST] [--port PORT] [--check] [--dry-run] [--skip-secrets] [{universal,all,claude-code,codex,openclaw}]` |
| `agent init` | Initialize agent onboarding files for Claude Code. | `rosclaw agent init [-h] [--project-root PROJECT_ROOT] [--profile PROFILE] [--robot ROBOT] [--transport {stdio,http,sse}] [--host HOST] [--port PORT] [--check] [--dry-run] [--skip-secrets] {claude-code}` |
| `agent doctor` | Diagnose ROSClaw agent onboarding configuration. | `rosclaw agent doctor [-h] [--project-root PROJECT_ROOT] [--skip-secrets] {universal,all,claude-code,codex,openclaw}` |
| `agent test` | Run onboarding and MCP tests for ROSClaw agent integrations. | `rosclaw agent test [-h] [--project-root PROJECT_ROOT] [--quick] [--verbose] [--mcp-probe] {universal,all,claude-code,codex,openclaw}` |

## Config commands

| Command | Description | Usage |
|---------|-------------|-------|
| `config show` | Show current config | `rosclaw config show [-h]` |
| `config path` | Show config file path | `rosclaw config path [-h]` |
| `config validate` | Validate config schema | `rosclaw config validate [-h]` |
| `config edit` | Open config in editor | `rosclaw config edit [-h] [--editor EDITOR]` |

## Profile commands

| Command | Description | Usage |
|---------|-------------|-------|
| `profile list` | List available profiles | `rosclaw profile list [-h]` |
| `profile use` | Activate a profile | `rosclaw profile use [-h] {offline,cloud,hybrid,ros2,sim}` |
| `profile current` | Show active profile | `rosclaw profile current [-h]` |

## Demo commands

| Command | Description | Usage |
|---------|-------------|-------|
| `demo mobile-pid` | Mobile base PID control demo | `rosclaw demo mobile-pid [-h] [--robot-id ROBOT_ID] [--target TARGET] [--kp KP] [--ki KI] [--kd KD] [--backend {mock,ros2}]` |
| `demo tabletop-grasp` | Tabletop grasp demo | `rosclaw demo tabletop-grasp [-h] [--robot-id ROBOT_ID] [--object OBJECT]` |

## Fleet commands

| Command | Description | Usage |
|---------|-------------|-------|
| `fleet status` | Show fleet status | `rosclaw fleet status [-h] [--workspace WORKSPACE] [--json]` |
| `fleet stop` | Broadcast emergency stop to all bodies | `rosclaw fleet stop [-h] [--workspace WORKSPACE] [--reason REASON]` |

## Ros commands

| Command | Description | Usage |
|---------|-------------|-------|
| `ros ping` | Ping rosbridge endpoint | `rosclaw ros ping [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json]` |
| `ros discover` | Discover ROS graph | `rosclaw ros discover [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json] [--out OUT]` |
| `ros compile` | Compile CapabilityManifest from ROS graph | `rosclaw ros compile [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json] [--output OUTPUT] [--graph GRAPH]` |
| `ros list-capabilities` | List compiled capabilities | `rosclaw ros list-capabilities [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json] [--manifest MANIFEST]` |
| `ros inspect-capability` | Inspect a capability | `rosclaw ros inspect-capability [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json] [--manifest MANIFEST] capability_id` |
| `ros validate-capability` | Validate args against safety contract | `rosclaw ros validate-capability [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json] [--manifest MANIFEST] [--args ARGS] capability_id` |
| `ros execute-capability` | Execute a capability | `rosclaw ros execute-capability [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json] [--manifest MANIFEST] [--args ARGS] [--dry-run] capability_id` |
| `ros emergency-stop` | Send emergency zero velocity | `rosclaw ros emergency-stop [-h] [--endpoint ENDPOINT] [--robot-id ROBOT_ID] [--json]` |
