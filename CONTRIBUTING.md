# Contributing to ROSClaw

Thank you for your interest in contributing to ROSClaw! This document outlines the development standards, PR process, and code style guidelines to ensure a smooth collaboration.

## Development Setup

### Prerequisites

- Python >= 3.11
- pip or uv
- Git

### Install Dependencies

```bash
git clone https://github.com/ros-claw/rosclaw.git
cd rosclaw
pip install -e ".[dev]"
```

Or use the Makefile:

```bash
make install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation updates
- `refactor/` for code refactoring

### 2. Make Changes

- Write code following the style guidelines below
- Add tests for new functionality
- Update documentation if needed

### 3. Run Tests and Linting

Before committing, ensure all checks pass:

```bash
make all    # Runs lint + test
```

Or run individually:

```bash
make lint   # ruff check
make test   # pytest
make format # ruff format
```

### 4. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new skill loader
fix: correct joint validation bounds
docs: update API reference
test: add coverage for firewall edge cases
refactor: simplify event bus routing
ci: update GitHub Actions workflow
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request against the `main` branch.

## PR Process

1. **Title**: Use Conventional Commit format (`feat:`, `fix:`, etc.)
2. **Description**: Explain what changed and why
3. **Tests**: All CI checks must pass (lint, type-check, test matrix)
4. **Review**: At least one maintainer approval required
5. **Merge**: Use the repository's configured merge strategy after approval

## Repository Hygiene

Read [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) before adding a
new top-level file or directory.

- Keep raw reports, command transcripts, datasets, keys, and local evidence
  outside Git.
- Put reviewed evidence summaries under `docs/evidence/`.
- Keep independently installable worker packages under `worker_plugins/`.
- Website database migrations belong only to `ros-claw/rosclaw-website`.
- Checked-in Agent instructions must not contain machine-specific absolute
  paths.

## Code Style

### Python

- **Formatter**: [ruff](https://docs.astral.sh/ruff/)
- **Line length**: 100 characters
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Use Google-style or plain descriptions for non-obvious behavior

### Key Rules

- **No comments explaining WHAT**: Well-named identifiers should do that
- **Comments for WHY only**: Hidden constraints, subtle invariants, workarounds
- **Validate at boundaries**: Check user input and external API responses; trust internal code
- **No premature abstractions**: Three similar lines are better than an early abstraction
- **Default to no comments**: If removing a comment wouldn't confuse a reader, don't write it

### Example

```python
def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
    """Command joints to target positions."""
    self._ensure_ready("move_joints")
    self._validate_joint_positions(positions)
    # ... implementation
```

## Architecture Principles

- **EventBus is the only channel**: All module communication goes through publish/subscribe. No direct calls between modules.
- **LifecycleMixin**: All components with initialization/shutdown logic must inherit from `LifecycleMixin` and implement `_do_initialize()` / `_do_stop()`.
- **Grounding Engines**: New features should fit into one of the six grounding engines (Physical, Action, Timeline, Experience, Skill, Collaboration).

## Testing

- **Framework**: pytest
- **Coverage**: Aim for high coverage on business logic; skip framework/type-system tests
- **Mocking**: Mock external dependencies (hardware, network), not internal modules
- **Integration tests**: Place in `tests/` with `test_*.py` naming

## Questions?

- Open an issue for bug reports or feature requests
- Join discussions in existing issues before starting major work

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
