# Repository Layout and Ownership

The repository root is a compatibility surface, not a general artifact
directory. Files remain at the root only when a tool, package manager, GitHub,
or an Agent framework discovers them there.

## Root contracts

These files intentionally remain at the root:

- `README.md`, `README.zh.md`, `LICENSE`, `CHANGELOG.md`, and
  `CONTRIBUTING.md` are public project and release metadata.
- `AGENTS.md`, `CLAUDE.md`, and `ROSCLAW.md` are distinct Agent-framework
  discovery entry points. Their checked-in commands must be repository
  relative; per-checkout absolute MCP commands belong only in ignored local
  configuration.
- `pyproject.toml`, `Dockerfile`, Compose files, `Makefile`, and `install.sh`
  are build and deployment entry points.

Do not replace the Agent entry points with symlinks. Agent harnesses differ in
how they resolve symlinks and referenced instruction files.

## Owned source trees

- `src/` owns runtime code.
- `tests/` owns executable verification.
- `worker_plugins/` owns independently installable packages that run in
  isolated worker environments. Plugin-specific policy fixtures live with the
  plugin that consumes them.
- `benchmarks/` owns reproducible benchmark contracts and runners.
- `docs/evidence/` owns reviewed, durable evidence summaries referenced by the
  canonical product status.

## External deployment ownership

The `ros-claw/rosclaw-website` repository is the sole owner of website
Supabase migrations. Do not copy those migrations into this runtime
repository: duplicated migration histories drift and can apply different
schemas under the same version number.

## Generated artifacts

Raw command transcripts, test runs, local reports, datasets, keys, receipts,
and hardware captures must stay outside Git. In particular:

- root `reports/` is ignored;
- local Agent/MCP configuration such as `.mcp.json` is ignored;
- runtime data belongs under `ROSCLAW_HOME` or another external artifact root;
- reviewed evidence is reduced to a portable summary under `docs/evidence/`.

Git history remains the archive for removed implementation reports. Do not
reintroduce machine paths, generated logs, or one-time handoff reports as
current documentation.
