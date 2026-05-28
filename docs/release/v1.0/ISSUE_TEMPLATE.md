# ROSClaw v1.0 Issue Template

Copy this template for every issue found during audit.

---

## ISSUE-XXX: [Short Title]

**Severity**: P0 / P1 / P2 / P3
**Module**: [module name]
**Owner**: [tmux session name]
**Detected by**: [auditor session name]
**Status**: open / fixed / verified / deferred

### Problem
[One-sentence description of the issue]

### Evidence
- file: [path]
- line: [number]
- direct call/import: [what violates the boundary]

### Why it matters
[Explain the architectural or functional impact]

### Expected behavior
[What should happen instead]

### Suggested fix
[Concrete fix recommendation]

### Verification
```bash
[Command or test that proves the fix works]
```

---

## Severity Definitions

- **P0**: Release blocker. System doesn't work or architecture boundary violated.
- **P1**: Should fix before release. Degraded functionality or engineering risk.
- **P2**: Defer to v1.1. Nice to have but not blocking.
- **P3**: Cosmetic or documentation improvement.

## Filing Rules

1. All issues go into `docs/release/v1.0/audits/audit-[module].md`
2. Use sequential numbering: ISSUE-001, ISSUE-002, ...
3. Cross-reference related issues across audit files
4. The integrator (rosclaw) consolidates into a master issue tracker
