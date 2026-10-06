# PI 1.0.4 compatibility

ROSClaw pins all four `@earendil-works/pi-*` dependencies and overrides to
1.0.4. The package check also verifies each installed package version, so a
manifest update alone cannot satisfy the pin check.

The [upstream changelog](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/CHANGELOG.md)
describes tool patterns, MCP selection and shutdown changes. ROSClaw continues
to install its own inline extension and select its model tools through the
public session API. Its optional execution budget uses the public `tool_call`
hook and restricts exact tool names.

Validation in an isolated worktree used packages from the official npm
registry, applied the existing upstream patches, built the distribution and
ran the full Node suite: 352 passed, 3 skipped. The first run's sole failure
was the old test pin of 1.0.3; it was updated and strengthened to check the
installed packages before the successful run. Python compilation, 121-file
core type checks and Practice tests also passed (183 passed, 9 skipped).
Existing whole-repository Ruff and formatting findings remain unchanged.

A real ROSClaw CLI/PI session with private dummy credentials and a synthetic
Kimi SSE response confirmed the output-limit notice, hidden reasoning,
preserved explicit retry zero, normal quit and unchanged global settings.
Socket restrictions and fake fetch prevented external network calls. This is
runtime compatibility evidence, not a paid-model or physical capability test.
Existing frozen experiments retain their recorded runtime versions.
