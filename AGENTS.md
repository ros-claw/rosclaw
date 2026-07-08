# ROSClaw Agent Instructions

ROSClaw is physical-AI runtime infrastructure. Treat robot, ROS, actuator,
motor, and hardware commands as safety-sensitive.

## Safety Boundary

- Do not publish ROS topics, actuate hardware, run real robot skills, or mutate
  a live robot workspace unless the user explicitly requests that exact action.
- Prefer fixture, mock, simulation, read-only, dry-run, or temporary
  `ROSCLAW_HOME` workflows for validation.
- For any motion-related request, validate through sandbox/firewall first and
  require operator confirmation before real execution.

## Useful Commands

```bash
python -m compileall -q src tests
ruff check .
ruff format --check .
mypy --config-file .github/mypy-ci.ini src/rosclaw/mcp/adapters src/rosclaw/mcp/onboarding src/rosclaw/core/runtime.py src/rosclaw/cli.py src/rosclaw/body src/rosclaw/firstboot src/rosclaw/hub
pytest tests/practice -q
```

Use the repo skill at `.agents/skills/rosclaw/SKILL.md` for deeper CLI,
Practice evidence, MCP, and agent-framework workflows.
