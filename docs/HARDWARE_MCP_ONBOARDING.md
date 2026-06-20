# Hardware MCP Onboarding

ROSClaw can automatically install, bind, and health-check hardware MCP servers from
a declarative manifest. A built-in offline registry ships with `unitree-g1` and
`realsense-d455` examples; a remote Hub client can be plugged in later.

## Commands

### Install a hardware MCP server

```bash
# Preview the plan without mutating the system
rosclaw mcp install unitree-g1 --dry-run --offline

# Machine-readable plan
rosclaw mcp install unitree-g1 --dry-run --offline --json

# Actually install
rosclaw mcp install unitree-g1 --offline
```

The install lifecycle:

1. Alias → canonical manifest ID resolution
2. Version solving (exact &gt; lockfile &gt; stable channel)
3. Effective permission check
4. Preflight shell commands
5. Artifact installation (Python package, Docker, or remote URL)
6. Runtime runner registration (`~/.rosclaw/mcp/runtime/<server>.yaml`)
7. `body.yaml` binding (e-URDF profile + binding key)
8. `.mcp.json` merge for Claude Code

### List available and installed servers

```bash
rosclaw mcp list --offline
rosclaw mcp list --offline --json
```

### Health-check a server

```bash
rosclaw mcp health unitree-g1
rosclaw mcp health unitree-g1 --json
rosclaw mcp health unitree-g1 --full   # includes hardware/safety checks
rosclaw mcp health                     # check every installed server
```

Check categories:

- `install` — manifest integrity, runtime config, runner executable, server dir
- `protocol` — transport command resolvable, optional `--full` MCP handshake
- `binding` — e-URDF profile installed and `body.yaml` binding key present
- `permissions` — required permissions granted, forbidden permissions not granted
- `agent` — managed server present in project `.mcp.json`
- `hardware` / `safety` — only with `--full`

## State files

| Path | Purpose |
|------|---------|
| `~/.rosclaw/mcp/installed.yaml` | Installed server registry |
| `~/.rosclaw/mcp/permissions.yaml` | Granted / denied / pending permission IDs |
| `~/.rosclaw/mcp/lock.yaml` | Locked manifest versions |
| `~/.rosclaw/mcp/runtime/<server>.yaml` | Per-server runtime config |
| `~/.rosclaw/mcp/bin/rosclaw-mcp-run` | Shared runner executable |
| `<project-root>/.mcp.json` | Claude Code MCP config (ROSClaw-managed) |

## Manifests

Manifests are dataclass-based schemas in `src/rosclaw/mcp/onboarding/schema.py`.
Key sections:

- `artifact` — what to install (Python package, Docker image, remote URL)
- `mcp` — transport and server parameters
- `permissions` — required and optional permission declarations
- `body_binding` — keys to write into `body.yaml`
- `eurdf` — required/optional e-URDF profiles
- `health` — checks to run
- `claude` — fragment to merge into `.mcp.json`

Golden examples live in `tests/mcp/fixtures/manifests/`.

## Testing

Run the MCP onboarding test suite:

```bash
python3 -m pytest tests/mcp -q
```

Expected result: `103 passed, 1 skipped`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Preflight fails | Ensure the manifest's preflight commands are available on PATH |
| Protocol check fails | Verify the transport command resolves and the runner script exists |
| Binding check fails | Link `body.yaml` and confirm it contains the required binding key |
| Agent check fails | Confirm `.mcp.json` contains the managed server key |
| Permission check fails | Re-run `rosclaw mcp install <server>` to grant required permissions |

## See also

- `docs/MCP_USAGE.md` — Chinese-language MCP usage guide
- `src/rosclaw/mcp/onboarding/` — implementation
- `tests/mcp/test_cli.py` — CLI-level contract tests
