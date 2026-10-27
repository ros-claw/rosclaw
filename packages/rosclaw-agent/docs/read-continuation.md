# Ordinary `read`: bounded UTF-8 continuation

The ordinary ROSClaw `read` tool now wraps the pinned PI 1.4 read definition in
`src/harness/pi/bounded-read.ts`. Main-session `customTools` registers it under
the existing name `read`; no new tool permission or Bash allowance is required.

The default mode keeps PI's line offsets, limits, image handling and ordinary
errors. It also keeps the legacy line mode's whole-file IO. When one line exceeds
50 KiB, the visible notice gives a supported byte-mode restart instead of a
Bash command. Restart from byte zero and discard any previously collected line
fragments before reconstructing the complete file.

```json
{"path":"large.json","mode":"bytes","maxBytes":32768}
```

In byte mode, `content[0]` is the source text payload. The next text block contains
range metadata and the next opaque cursor, so it is visible to the model. Keep
that metadata separate from the payload. Continue on the same path and tool
instance with `mode: "bytes"` and the returned `cursor` until `eof: true`.
`maxBytes` is an integer from 4 through 32768; chunks end on UTF-8 codepoint
boundaries. Empty files and repeated EOF have explicit zero-byte ranges.

Cursors bind the resolved path, byte position and file revision. Ordinary local
edits or pathname replacements cause stale-cursor rejection; restart without a
cursor and discard earlier chunks. This is metadata-based change detection, not
a filesystem lock or protection from a filesystem that lies about its metadata.
Malformed cursors, unknown arguments, mixed line/byte arguments and invalid UTF-8
are rejected rather than silently coerced into progress. Binary files are outside
this mode's contract.

Existing remote `ReadOperations` continue to handle line/image requests. Byte
mode requires the explicit `BoundedReadOperationsV1` range seam; without it, the
remote request is blocked with `BYTE_MODE_UNSUPPORTED`, never silently read from
the local filesystem.

## Validation and provenance

The implementation and its regression tests were authored by ROSClaw Native.
Root's generic work only moved the complete module into the existing PI harness
boundary, preserving its bytes; Native updated the three import/regex identities.
No new `HP2-COMPAT` exception was added. No dependency was installed or modified.

Independent validation of the final source snapshot completed:

- TypeScript build: exit 0.
- Focused read/filetool/policy/CLI-policy/HP2 checks: 107 passed, no skips.
- Complete Node suite: 568 tests, 565 passed, no failures, 3 existing bwrap
  environment skips. These skips are not passing sandbox evidence.

Coverage includes the thirteen original huge-line cases, 679452-byte JSON,
51200/51201-byte boundaries, UTF-8/BOM/CRLF/empty tails, EOF, stale revisions,
real guarded edits, cancellation, parallel reads, remote seams, default images,
and read-only/exact-selftest policies. A separate offline official PI session
confirmed that the same-name custom definition replaces the built-in active
`read`; that routing probe did not execute a policy hook or contact a model.

Historical Native author U1 protocol/metadata failures and earlier compilation,
notice-test and HP2-structure failures remain recorded as failures. Independent
software checks do not regrade those author deliveries, assert OS sandboxing,
or grant SDK, hardware or physical-task qualification.
