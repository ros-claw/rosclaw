# Package release

Status: PASS.

Only the two authorized packages were published. Core was not published.

| Package | Version | PyPI | Git tag | Merge |
|---|---:|---|---|---|
| rosclaw-know | 1.3.0 | https://pypi.org/project/rosclaw-know/1.3.0/ | `v1.3.0` | `3e15dfb` / PR #6 |
| rosclaw-how | 1.3.0 | https://pypi.org/project/rosclaw-how/1.3.0/ | `v1.3.0` | `9b866d4` / PR #4 |

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `rosclaw_know-1.3.0-py3-none-any.whl` | `ae0a418a1f574449a556b0b17fe9064eb0ba0cff270844b5e514c5f59513ea4d` |
| `rosclaw_know-1.3.0.tar.gz` | `16d46d53ce980bc9b619c03eb2cea41b72a4041784adbf17e3327a64024749fb` |
| `rosclaw_how-1.3.0-py3-none-any.whl` | `bc4ab9ee22b8050ab27f9a8a25d4c78ae1e0966a5df78cee34cd1957ceb2d34a` |
| `rosclaw_how-1.3.0.tar.gz` | `42fde979567a17eccdba9a397af85553dbe1009f5b29f3127bf43db494262155` |

Both wheel and sdist passed `twine check`. PyPI JSON returned the same four
hashes. Fresh PyPI-hosted wheels were installed at exact version; service
imports and CLIs passed. Know reported all seven packaged migrations. Package
uninstall/reinstall was also verified.

Mode evidence:

- Know memory mode and doctor: PASS;
- Know SeekDB server mode and native tests: PASS;
- How service/API import and CLI: PASS;
- unavailable Know / disabled integration: PASS through explicit How
  abstention and Core disabled-state tests;
- embedded compatibility: covered by the existing full Know suite; this round
  did not claim an additional embedded-server performance result.
