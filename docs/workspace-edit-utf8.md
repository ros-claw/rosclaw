# ADR: Workspace file-tool UTF-8/empty-oldText repair (WORKSPACE_EDIT_UTF8_5250642c49f146d6b81cecd689bcecc9)

## Scope
Software-only repair of two custom workspace tool/test sources:
- `packages/rosclaw-agent/src/tools/workspace-pack.ts`
- `packages/rosclaw-agent/test/filetool-diagnostics.test.ts`

No model, hardware, ROS, or NN changes. Method signatures and tool surfaces remain compatible.

## Bugs fixed

1. **`edit` accepted empty `oldText`.** `text.split("")` counts N+1 occurrences, and an
   empty pattern can bypass the exactly-once check and insert `newText` into the file
   (counterexample: file `"ab"` became `"Xab"`, isError=false).
   Fix: reject `oldText.length === 0` after reading but **before any write-back**, with
   typed code `EDIT_OLD_TEXT_EMPTY` and the standard path diagnostics. File is untouched.

2. **`write` reported UTF-16 code units, not UTF-8 bytes.** `"汉字"` was reported as
   `2 bytes` while the on-disk file is 6 bytes.
   Fix: report `Buffer.byteLength(String(params.content), "utf-8")`.

## Preserved behavior (no concurrency failure invented)
- Typed path/admission failures (`PATH_SCOPE_DENIED`, `EDIT_OLD_TEXT_NOT_FOUND`,
  `EDIT_OLD_TEXT_NOT_UNIQUE`) unchanged.
- Legitimate multiple completed effects (two same-file non-overlapping edits;
  disjoint-file edits) remain fully supported.

## Tests added (`filetool-diagnostics.test.ts`)
- `empty oldText is rejected before any mutation; file unchanged`
- `write reports UTF-8 byte length, not UTF-16 code units`
- Existing diagnostics and extra-root tests unchanged and still passing.
