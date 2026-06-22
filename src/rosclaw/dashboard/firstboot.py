"""Dashboard firstboot page helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import is_workspace_initialized, load_install_state, resolve_home


def get_firstboot_state(home: Path | str | None = None) -> dict[str, Any]:
    """Return current First Boot state for the dashboard.

    Reads install metadata and configuration file presence from the workspace.
    """
    home_path = Path(home) if home else resolve_home()
    state: dict[str, Any] = {
        "home": str(home_path),
        "workspace_exists": False,
        "initialized": False,
        "install_state": None,
        "rosclaw_yaml_exists": False,
        "mcp_json_exists": False,
        "telemetry_yaml_exists": False,
        "steps": {},
    }

    try:
        state["workspace_exists"] = home_path.exists()
        state["initialized"] = is_workspace_initialized(home_path)
        state["install_state"] = load_install_state(home_path)
        state["rosclaw_yaml_exists"] = (home_path / "config" / "rosclaw.yaml").exists()
        state["mcp_json_exists"] = (home_path / "config" / "mcp.json").exists()
        state["telemetry_yaml_exists"] = (home_path / "config" / "telemetry.yaml").exists()
    except Exception as exc:  # noqa: BLE001
        state["error"] = str(exc)

    install = state["install_state"] or {}
    state["steps"] = {
        "bootstrap": bool(install.get("installed_at")),
        "workspace": state["initialized"],
        "config": state["rosclaw_yaml_exists"],
        "mcp": state["mcp_json_exists"],
        "telemetry": state["telemetry_yaml_exists"],
        "firstboot": bool(install.get("firstboot_completed")),
        "doctor": install.get("last_doctor_status") == "ready",
    }

    return state


def build_firstboot_command(choices: dict[str, Any]) -> str:
    """Build the equivalent non-interactive `rosclaw firstboot` CLI command."""
    cmd = ["rosclaw", "firstboot", "--yes"]

    profile = choices.get("profile", "offline")
    cmd.extend(["--profile", profile])

    robot = choices.get("robot", "sim_ur5e")
    cmd.extend(["--robot", robot])

    safety = choices.get("safety", "strict")
    cmd.extend(["--safety", safety])

    if choices.get("telemetry", False):
        cmd.append("--telemetry")
    else:
        cmd.append("--no-telemetry")

    if choices.get("mcp", True):
        cmd.append("--enable-mcp")
    else:
        cmd.append("--disable-mcp")

    if choices.get("sandbox", True):
        cmd.append("--enable-sandbox")
    else:
        cmd.append("--disable-sandbox")

    if choices.get("ros2", False):
        cmd.append("--enable-ros2")
    if choices.get("memory", False):
        cmd.append("--enable-memory")
    if choices.get("practice", False):
        cmd.append("--enable-practice")
    if choices.get("auto", False):
        cmd.append("--enable-auto")

    return " ".join(cmd)


def preview_firstboot_config(choices: dict[str, Any]) -> dict[str, Any]:
    """Return a human-readable preview of the configuration implied by choices."""
    return {
        "profile": choices.get("profile", "offline"),
        "robot": choices.get("robot", "sim_ur5e"),
        "safety": choices.get("safety", "strict"),
        "telemetry": bool(choices.get("telemetry", False)),
        "mcp": bool(choices.get("mcp", True)),
        "sandbox": bool(choices.get("sandbox", True)),
        "use_cases": {
            "ros2": bool(choices.get("ros2", False)),
            "memory": bool(choices.get("memory", False)),
            "practice": bool(choices.get("practice", False)),
            "auto": bool(choices.get("auto", False)),
        },
    }


FIRSTBOOT_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ROSClaw First Boot</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --danger: #ef4444; }
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 2rem; background: var(--bg); color: var(--text); line-height: 1.5; }
    h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
    h2 { font-size: 1.25rem; margin-top: 0; }
    .subtitle { color: var(--muted); margin-bottom: 1.5rem; }
    .card { background: var(--card); border-radius: 0.5rem; padding: 1.25rem; margin: 1rem 0; }
    .muted { color: var(--muted); }
    .status { font-weight: bold; }
    .status.ok { color: var(--ok); }
    .status.pending { color: var(--warn); }
    .status.error { color: var(--danger); }
    ul.steps { list-style: none; padding: 0; }
    ul.steps li { display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #334155; }
    ul.steps li:last-child { border-bottom: none; }
    label { display: block; margin: 0.75rem 0 0.25rem; color: var(--muted); }
    select, input[type="text"] { width: 100%; padding: 0.5rem; border-radius: 0.25rem; border: 1px solid #475569; background: #0f172a; color: var(--text); box-sizing: border-box; }
    .checks { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; margin-top: 0.5rem; }
    .checks label { display: flex; align-items: center; gap: 0.5rem; margin: 0; color: var(--text); cursor: pointer; }
    .checks input { accent-color: var(--accent); }
    code { display: block; background: #0f172a; padding: 0.75rem; border-radius: 0.25rem; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
    .actions { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
    button { background: var(--accent); color: #0f172a; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; font-weight: bold; }
    button:hover { opacity: 0.9; }
    button.secondary { background: #334155; color: var(--text); }
    .notice { border-left: 4px solid var(--accent); padding-left: 0.75rem; color: var(--muted); }
  </style>
</head>
<body>
  <h1>ROSClaw First Boot</h1>
  <div class="subtitle">Visual setup wizard — no robot will be moved.</div>

  <div class="card">
    <h2>Current Status</h2>
    <div id="status">Loading...</div>
    <ul class="steps" id="steps"></ul>
    <div class="actions">
      <button onclick="refreshStatus()">Refresh Status</button>
      <button class="secondary" onclick="window.open('/body', '_blank')">Body Dashboard</button>
    </div>
  </div>

  <div class="card">
    <h2>Configure</h2>
    <form id="firstboot-form">
      <label for="profile">Operating profile</label>
      <select id="profile" name="profile">
        <option value="offline" selected>Offline (default, no cloud)</option>
        <option value="hybrid">Hybrid</option>
        <option value="cloud">Cloud</option>
      </select>

      <label for="robot">Default robot</label>
      <select id="robot" name="robot">
        <option value="sim_ur5e" selected>sim_ur5e</option>
        <option value="turtlebot">turtlebot</option>
        <option value="unitree_go2">unitree_go2</option>
        <option value="unitree_g1">unitree_g1</option>
      </select>

      <label for="safety">Safety level</label>
      <select id="safety" name="safety">
        <option value="strict" selected>strict</option>
        <option value="moderate">moderate</option>
        <option value="relaxed">relaxed</option>
      </select>

      <label>Modules</label>
      <div class="checks">
        <label><input type="checkbox" id="sandbox" name="sandbox" checked> Sandbox</label>
        <label><input type="checkbox" id="mcp" name="mcp" checked> MCP / Claude Code</label>
        <label><input type="checkbox" id="ros2" name="ros2"> ROS 2</label>
        <label><input type="checkbox" id="memory" name="memory"> Memory</label>
        <label><input type="checkbox" id="practice" name="practice"> Practice</label>
        <label><input type="checkbox" id="auto" name="auto"> Auto evolution</label>
        <label><input type="checkbox" id="telemetry" name="telemetry"> Anonymous telemetry</label>
      </div>
    </form>
  </div>

  <div class="card">
    <h2>Run in Terminal</h2>
    <p class="muted">Copy this command and run it in your terminal to apply the configuration.</p>
    <code id="command">rosclaw firstboot --yes --profile offline --robot sim_ur5e --safety strict --no-telemetry --enable-mcp --enable-sandbox</code>
    <div class="actions">
      <button onclick="copyCommand()">Copy Command</button>
    </div>
    <p class="notice" style="margin-top: 1rem;">
      The dashboard wizard is read-only for safety. After running the command, click <strong>Refresh Status</strong> to see the updated state.
    </p>
  </div>

  <script>
    function statusClass(done) {
      return done ? 'ok' : 'pending';
    }

    async function refreshStatus() {
      try {
        const res = await fetch('/api/firstboot');
        const data = await res.json();
        const install = data.install_state || {};
        const completed = install.firstboot_completed;
        document.getElementById('status').innerHTML =
          `<div>Workspace: <strong>${data.home}</strong></div>` +
          `<div>Initialized: <span class="status ${statusClass(data.initialized)}">${data.initialized ? 'yes' : 'no'}</span></div>` +
          `<div>Firstboot completed: <span class="status ${statusClass(completed)}">${completed ? 'yes' : 'no'}</span></div>` +
          `<div class="muted">Install version: ${install.installer_version || 'unknown'} · Channel: ${install.install_channel || 'unknown'}</div>`;

        const steps = data.steps || {};
        const list = document.getElementById('steps');
        list.innerHTML = Object.entries(steps).map(([key, done]) =>
          `<li><span>${key}</span><span class="status ${statusClass(done)}">${done ? 'done' : 'pending'}</span></li>`
        ).join('');
      } catch (err) {
        document.getElementById('status').innerHTML = `<span class="status error">Error: ${err.message}</span>`;
      }
    }

    async function updatePreview() {
      const choices = {
        profile: document.getElementById('profile').value,
        robot: document.getElementById('robot').value,
        safety: document.getElementById('safety').value,
        sandbox: document.getElementById('sandbox').checked,
        mcp: document.getElementById('mcp').checked,
        ros2: document.getElementById('ros2').checked,
        memory: document.getElementById('memory').checked,
        practice: document.getElementById('practice').checked,
        auto: document.getElementById('auto').checked,
        telemetry: document.getElementById('telemetry').checked,
      };
      try {
        const res = await fetch('/api/firstboot/preview', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(choices),
        });
        const data = await res.json();
        document.getElementById('command').textContent = data.command || '';
      } catch (err) {
        document.getElementById('command').textContent = `Error generating preview: ${err.message}`;
      }
    }

    async function copyCommand() {
      const text = document.getElementById('command').textContent;
      try {
        await navigator.clipboard.writeText(text);
        const btn = document.querySelector('button[onclick="copyCommand()"]');
        const old = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = old, 1500);
      } catch (err) {
        alert('Copy failed: ' + err.message);
      }
    }

    document.getElementById('firstboot-form').addEventListener('change', updatePreview);
    refreshStatus();
    updatePreview();
  </script>
</body>
</html>
"""
