# Independent review of declared artifact validation

The Native implementation adds opt-in local `schema_path` validation before
artifact registration. Supported keyword shapes, bounded reads and deep JSON
rejections passed 50 independent controls through the compiled tool and private
RPC bridge. Integration passed 456 Node tests (3 existing skips), 44 Python
bridge tests, 183 Practice tests (9 skips), and the required CI mypy scope.

The original Native episode remains failed: its check and regression commands
exited successfully, but guarded Bash truncated their structured output at
64 KiB. The Native evidence report records its own declared results; use
[the independent review receipt](../reports/declared-artifact-schema-root-review.json)
for independently reproduced component results. Missing original output was
not reconstructed, and the whole episode was not rescored.

The four source and test files, Native ADR and Native evidence report were
copied without editing their bytes. AST comparison found the existing
registration body unchanged after the new opt-in prefix and all other
existing functions unchanged.

Validation covers the bounded local subset on files read for validation. It
does not guarantee a concurrent-file snapshot, general JSON Schema compliance
or physical execution. The separately tested supplied-perception adapter is
held for a newly found direct Python diagnostic boundary and is excluded here.
