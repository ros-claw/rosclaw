# Explicit manual release

## Status and historical failure

SOURCE_AUTHORED_NOT_TESTED. Preparation is not release authorization. The author
used read/edit only: no tests, build, processes, CI triggers, uploads, or
publication were executed. Root must review and run local regressions before
considering harmless CI; publication needs independent explicit authorization.

The original PR721 failure remains a FAILURE, not repaired execution evidence:
on baseline `e648d6d66d3a5ff7710c4b1fa33a48a26e99d893`, ordinary manual run
39064924663 unexpectedly entered release. Job 117495606963 failed at
`Record artifact digests (traceability)` because `sha256sum dist/*` treated the
`dist/js-stage` directory as a file. Root reports the other build/integration/
regression/lint tests passed. Both TestPyPI and PyPI steps were SKIPPED; nothing
was published. This source change neither reruns nor overwrites that history.

## Release admission

`workflow_dispatch.inputs.publish_testpypi` is a boolean, optional, default false.
Ordinary manual CI with the input absent or false still has build, integration,
and test jobs available but skips the entire release job. Only the typed boolean
true opts into the existing TestPyPI rehearsal. Both the release admission and
the TestPyPI step check `toJSON(inputs.publish_testpypi) == 'true'`: boolean true
serializes to `true`, while string `true` serializes with quotes, and strings
`false`, numbers, empty strings, and null do not opt in. This is not a check of
the truthiness of `github.event.inputs`, whose strings can mislead.

The existing push + `refs/tags/v` release predicate and the existing push-only
PyPI step are preserved. Effective PyPI admission remains their conjunction;
branch pushes and PRs cannot publish even with an input set. No tag trigger was
added: `on.push` still declares only main/develop branches, so this retained
predicate does not itself enable tag-triggered runs. Environments, permissions,
action identities, dependencies, release step names/order/count, and other jobs
are unchanged. Manual opt-in selects TestPyPI, not PyPI, even on a tag ref.

## Exact artifact digest policy

The existing digest step now validates before emitting any digest records:

- `dist` must be a real directory, not a symlink.
- Exactly one `rosclaw-VERSION-py3-none-any.whl` and one
  `rosclaw-VERSION.tar.gz` must be regular, non-symlink files. This is the
  existing pure-Python wheel naming contract; arbitrary wheel tags or projects
  are not accepted. Versions must match exactly and use normalized numeric
  release components, optional a/b/rc, post/dev, and optional local components.
- Missing, duplicate/ambiguous, malformed, foreign, hidden, extra, directory,
  symlink (including dangling), and special-file artifacts fail closed.
- The sole optional extra entry is the exact name `js-stage`, and only as a real
  directory. It is build support, not a standalone distribution: it is neither
  hashed nor emitted as a file for publication. Its tree may contain only real
  directories and regular files; symlink/special lookalikes and traversal
  errors are rejected. Content/package validity is not established by this
  structural check. Existing build resource checks remain responsible for the
  staging inputs embedded in the distributions.
- Only the validated pair is streamed read-only into SHA-256, ordered by
  filename, with standard `sha256  dist/filename` traceability lines. No rebuild,
  archive mutation, staging removal, helper production file, or new publishing
  step is introduced. These are the same downloaded built bytes. The directory
  stays build support; only the wheel/sdist are distribution-file candidates for
  the unchanged publisher action.

## Authored regression coverage (not executed)

`tests/test_ci_release_dispatch.py` reads the actual YAML with BaseLoader so
`on` is not coerced to a YAML 1.1 boolean. Input defaults/types are checked
separately. A deliberately bounded expression evaluator interprets the actual
job and publisher conditions, rejecting unsupported syntax. Its event/ref/input
matrix covers PRs, branch pushes, qualifying/nonqualifying tags, and dispatch
with absent/null, false, true, misleading strings, numbers, and containers. This
is local semantic regression coverage, not evidence of a live GitHub run.

Local digest tests extract the entire actual revised YAML `run` command and
execute only that command in temporary workspaces using bash with a ten-second
subprocess timeout. Positive fixtures include real JS staging directories and
independently computed hashes, ordering, and unchanged-byte checks. Negative
fixtures cover missing/extra/ambiguous/malformed distributions, mismatched
versions, symlinks, directories, FIFOs, abnormal staging entries, and a missing
or symlinked dist root. Wheel/sdist fixture payloads are dummy bytes: they claim
hashing and selection coverage only, no wheel build or package validation
credit. No publisher, upload, build script, full-release test, SDK, or network
operation is invoked by these tests. Root, not the author, runs them after
review. SOURCE_AUTHORED_NOT_TESTED STOP.
