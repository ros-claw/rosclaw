/** ROSClaw / PI 1.4 supported integration wrapper, not an upstream patch.
 * Lines/images deliberately delegate verbatim (including full-file line IO).
 * Bytes use bounded positional IO, not readFile, JSON parsing or a cache.
 * Cursor scope: this tool instance, resolved path, file revision, byte offset.
 * Local revisions use dev/ino/size/mtimeNs/ctimeNs, checked on the handle AND
 * pathname before/after every chunk. This detects ordinary edits/replacements,
 * not a malicious filesystem lying about metadata; no filesystem lock is claimed.
 */
import { constants } from "node:fs";
import { open, stat } from "node:fs/promises";
import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { Type } from "@earendil-works/pi-ai";
import type { Static } from "typebox";
import {
 createReadToolDefinition, defineTool, type ReadToolOptions, type ReadToolDetails,
} from "@earendil-works/pi-coding-agent";

// PI 1.4 path semantics (~, @ prefix, Unicode spaces, macOS normalization).
// Resolve beside the pinned package entry, also when running from dist/.
const { resolveReadPathAsync } = await import(new URL("./core/tools/path-utils.js",
 import.meta.resolve("@earendil-works/pi-coding-agent")).href) as {
 resolveReadPathAsync(path: string, cwd: string): Promise<string>;
};

export const BYTE_CHUNK_MAX = 32 * 1024;
const SCHEMA = "rosclaw.read.bytes.v1";
const CURSOR_RECOVERY = "Discard previous chunks. Restart the intended path with mode=bytes, omitting cursor, offset and limit; then use only its newly returned nextCursor.";
const parameters = Type.Object({
 path: Type.String({ description: "File path (same semantics as line read)" }),
 mode: Type.Optional(Type.Union([Type.Literal("lines"), Type.Literal("bytes")])),
 offset: Type.Optional(Type.Number({ description: "Line offset, 1-indexed; lines mode only" })),
 limit: Type.Optional(Type.Number({ description: "Line limit; lines mode only" })),
 cursor: Type.Optional(Type.String({ maxLength: 2048, description: "Opaque nextCursor from bytes mode; omit to start at byte 0" })),
 maxBytes: Type.Optional(Type.Integer({ minimum: 4, maximum: BYTE_CHUNK_MAX,
  description: "Bytes mode payload cap; default 32768, cut only at UTF-8 codepoint boundaries" })),
}, { additionalProperties: false });

/** Versioned remote range seam. Revisions MUST change on any content change or
 * pathname replacement, including same-sized edits, and check the opened object
 * and the current pathname. readAt must return <= length bytes from that handle.
 * Old ReadOperations are still delegated unchanged for lines/images; if supplied
 * without this v1 seam, bytes fail closed (never silently fall back to local IO).
 */
export interface BoundedReadOperationsV1 {
 version: 1;
 open(path: string, signal?: AbortSignal): Promise<{
  size: number;
  revision(): Promise<string>;
  readAt(offset: number, length: number, signal?: AbortSignal): Promise<Buffer>;
  close(): Promise<void>;
 }>;
}
export interface BoundedReadOptions extends ReadToolOptions {
 boundedOperations?: BoundedReadOperationsV1;
}
function fail(code: string, message: string): never {
 throw new Error(`${code}: ${message}`);
}
function cursorFailure(code: "INVALID_CURSOR" | "STALE_CURSOR", message: string): never {
 return fail(code, `${message}. ${CURSOR_RECOVERY}`);
}
/** PI strict provider schemas represent omitted optionals as null. PI's own
 * validateToolArguments normalizes those against the ORIGINAL schema, but runs
 * AFTER prepareArguments. Mirror only that omission rule here, before mode/type
 * checks, and at direct execute too. All five optional fields exclude null in
 * our schema: mode:null defaults to lines; cursor:null starts at byte zero;
 * maxBytes:null uses the default cap; inactive null fields cause no conflict.
 * Required path and unknown keys are never normalized. Non-null values are not
 * coerced. Copy so provider arguments (including frozen objects) stay intact.
 */
function prepareReadArguments(raw: unknown): Static<typeof parameters> {
 if (!raw || typeof raw !== "object" || Array.isArray(raw)) fail("INVALID_READ_ARGUMENTS", "expected an object");
 const input = { ...raw } as Record<string, unknown>;
 for (const key of ["mode", "offset", "limit", "cursor", "maxBytes"]) {
  if (input[key] === null) delete input[key];
 }
 if (Object.keys(input).some(k => !["path", "mode", "offset", "limit", "cursor", "maxBytes"].includes(k))
  || typeof input.path !== "string"
  || ("mode" in input && input.mode !== "lines" && input.mode !== "bytes")) {
  fail("INVALID_READ_ARGUMENTS", "unknown keys or invalid path/mode; use mode=lines|bytes");
 }
 const bytes = input.mode === "bytes";
 if (bytes ? ("offset" in input || "limit" in input) : ("cursor" in input || "maxBytes" in input)) {
  fail("READ_MODE_CONFLICT", "lines uses offset/limit; bytes uses cursor/maxBytes");
 }
 if (bytes && (("maxBytes" in input && (typeof input.maxBytes !== "number"
  || !Number.isSafeInteger(input.maxBytes) || input.maxBytes < 4 || input.maxBytes > BYTE_CHUNK_MAX))
  || ("cursor" in input && (typeof input.cursor !== "string" || !input.cursor.length || input.cursor.length > 2048)))) {
  fail("INVALID_READ_ARGUMENTS", "bytes needs maxBytes integer 4..32768 and a nonempty returned cursor");
 }
 // Preserve PI's legacy Number ranges (including fractional/zero/negative line
 // values), but reject strings/booleans before PI can coerce them to numbers.
 if (!bytes) for (const key of ["offset", "limit"]) {
  if (input[key] !== undefined && (typeof input[key] !== "number" || !Number.isFinite(input[key]))) {
   fail("INVALID_READ_ARGUMENTS", `${key} must be a finite number`);
  }
 }
 return input as Static<typeof parameters>;
}
function checkAbort(signal?: AbortSignal) {
 if (signal?.aborted) throw new Error("Operation aborted");
}
// Abort promptly even when a legacy/remote operation ignores the signal. The
// work still owns its finally/close and cannot publish a late successful result.
async function cancellable<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T> {
 checkAbort(signal);
 let abort!: () => void;
 const cancelled = new Promise<never>((_resolve, reject) => {
  abort = () => reject(new Error("Operation aborted"));
  signal?.addEventListener("abort", abort, { once: true });
 });
 try { return await Promise.race([work(), cancelled]); }
 finally { signal?.removeEventListener("abort", abort); }
}
const hash = (text: string) => createHash("sha256").update(text).digest("hex");
const localRanges: BoundedReadOperationsV1 = {
 version: 1,
 async open(path, signal) {
  checkAbort(signal);
  // NONBLOCK avoids hanging on a FIFO before the regular-file check.
  const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
   checkAbort(signal);
   const initial = await file.stat({ bigint: true });
   if (!initial.isFile()) fail("BYTE_MODE_UNSUPPORTED", "bytes mode requires a regular file");
   if (initial.size > BigInt(Number.MAX_SAFE_INTEGER)) fail("BYTE_MODE_UNSUPPORTED", "file size is not a safe byte integer");
   const fingerprint = (s: typeof initial) => [s.dev, s.ino, s.size, s.mtimeNs, s.ctimeNs].join(":");
   return {
    size: Number(initial.size),
    async revision() {
     const handleState = await file.stat({ bigint: true });
     let pathState: typeof initial;
     try { pathState = await stat(path, { bigint: true }); }
     catch { return cursorFailure("STALE_CURSOR", "file pathname changed"); }
     if (fingerprint(handleState) !== fingerprint(initial)
      || fingerprint(handleState) !== fingerprint(pathState)) {
      cursorFailure("STALE_CURSOR", "file replaced during read");
     }
     return fingerprint(handleState);
    },
    async readAt(offset, length, readSignal) {
     const buffer = Buffer.alloc(length);
     let used = 0;
     while (used < length) {
      checkAbort(readSignal);
      const { bytesRead } = await file.read(buffer, used, length - used, offset + used);
      if (!bytesRead) break;
      used += bytesRead;
     }
     checkAbort(readSignal);
     return buffer.subarray(0, used);
    },
    close: () => file.close(),
   };
  } catch (error) { await file.close(); throw error; }
 },
};

/** The exact definition registered by main customTools; SDK0 tests wrap this
 * same definition with PI's actual definition-to-AgentTool adapter. No new name,
 * permission table, scheduler flag, SDK/model call or effect admission is added.
 */
export function buildBoundedReadTool(cwd: string, options: BoundedReadOptions = {}) {
 const legacy = createReadToolDefinition(cwd, options);
 const ranges = options.boundedOperations ?? (options.operations ? undefined : localRanges);
 if (ranges && ranges.version !== 1) fail("BYTE_MODE_UNSUPPORTED", "unsupported range operations version");
 const secret = randomBytes(32);
 const sign = (data: string) => createHmac("sha256", secret).update(data).digest();
 const encode = (path: string, revision: string, offset: number) => {
  const data = Buffer.from(JSON.stringify([1, hash(path), hash(revision), offset])).toString("base64url");
  return `${data}.${sign(data).toString("base64url")}`;
 };
 const decode = (cursor: string, path: string, revision: string, size: number) => {
  const parts = cursor.split(".");
  if (parts.length !== 2 || !parts.every(p => /^[A-Za-z0-9_-]+$/.test(p))) cursorFailure("INVALID_CURSOR", "malformed cursor");
  const signature = Buffer.from(parts[1], "base64url");
  if (signature.length !== 32 || !timingSafeEqual(sign(parts[0]), signature)) {
   cursorFailure("INVALID_CURSOR", "cursor not issued by this read instance");
  }
  let value: unknown;
  try { value = JSON.parse(Buffer.from(parts[0], "base64url").toString("utf8")); }
  catch { return cursorFailure("INVALID_CURSOR", "malformed cursor payload"); }
  if (!Array.isArray(value) || value.length !== 4 || value[0] !== 1
   || value[1] !== hash(path) || !Number.isSafeInteger(value[3]) || value[3] < 0) {
   cursorFailure("INVALID_CURSOR", "cursor path, version or byte position mismatch");
  }
  if (value[2] !== hash(revision)) cursorFailure("STALE_CURSOR", "file changed");
  if (value[3] > size) cursorFailure("INVALID_CURSOR", "cursor beyond EOF");
  return value[3] as number;
 };
 // The legacy output schema is string/image only. Do not falsely advertise it
 // for bytes metadata; line structuredContent still comes from PI unchanged.
 const { outputSchema: _legacyOutputSchema, ...base } = legacy;
 return defineTool<typeof parameters, (ReadToolDetails & { boundedRead?: object; continuation?: object }) | undefined>({
  ...base,
  parameters,
  description: legacy.description + " ROSClaw PI 1.4 wrapper: for huge lines, restart with mode='bytes' (no offset/limit). "
   + "Returns exact UTF-8 source prefix plus metadata; for a complete file, use nextCursor as cursor on the same path until eof=true. "
   + "maxBytes=4..32768 (default 32768); cursor is byte-based, instance/revision-bound. "
   + "Optional null fields mean omitted (mode defaults to lines; cursor starts at byte 0). "
   + "No shell needed. Unknown keys/mixed modes are rejected. Invalid UTF-8 is explicitly blocked.",
  prepareArguments(raw) { return prepareReadArguments(raw); },
  async execute(id, params, signal, onUpdate, ctx) {
   // Use the same normalized copy for direct execution and normal PI dispatch.
   const input = prepareReadArguments(params);
   const byteMode = input.mode === "bytes";
   checkAbort(signal);
   if (!byteMode) {
    const result = await legacy.execute(id, { path: input.path, offset: input.offset, limit: input.limit }, signal, onUpdate, ctx);
    if (result.details?.truncation?.firstLineExceedsLimit) {
     const nextRead = ranges ? { path: input.path, mode: "bytes" as const } : null;
     const continuation = { schema: SCHEMA, status: ranges ? "restart_bytes" : "blocked",
      code: ranges ? "LINE_EXCEEDS_LIMIT" : "BYTE_MODE_UNSUPPORTED", nextRead };
     const totalBytes = result.details.truncation.totalBytes;
     const sizeHint = `File size: ${totalBytes} bytes; about ${Math.ceil(totalBytes / BYTE_CHUNK_MAX)} byte pages at default maxBytes=${BYTE_CHUNK_MAX} (UTF-8 boundaries can add pages). `;
     const text = ranges
      ? `[Line exceeds the line-read byte limit. ${sizeHint}Do not skip this line. Restart from file byte 0 with read ${JSON.stringify(nextRead)}. If you need the complete file, concatenate only bytes-mode payloads until eof=true; otherwise inspect only the data needed.]`
      : "[BYTE_MODE_UNSUPPORTED: LINE_EXCEEDS_LIMIT: remote ReadOperations has no BoundedReadOperationsV1. Byte continuation blocked; configure the versioned range adapter. Do not skip this line.]";
     return { ...result, content: [{ type: "text" as const, text }], structuredContent: text,
      details: { ...result.details, continuation } };
    }
    return result;
   }
   const maxBytes = input.maxBytes ?? BYTE_CHUNK_MAX;
   if (typeof maxBytes !== "number" || !Number.isSafeInteger(maxBytes) || maxBytes < 4 || maxBytes > BYTE_CHUNK_MAX
    || (input.cursor !== undefined && (typeof input.cursor !== "string" || !input.cursor.length || input.cursor.length > 2048))) {
    fail("INVALID_READ_ARGUMENTS", "bytes needs maxBytes integer 4..32768 and a nonempty returned cursor");
   }
   if (!ranges) fail("BYTE_MODE_UNSUPPORTED", "remote ReadOperations requires BoundedReadOperationsV1; no local fallback");
   return cancellable(async () => {
    const path = await resolveReadPathAsync(input.path, ctx?.cwd || cwd);
    checkAbort(signal);
    const reader = await ranges.open(path, signal);
    try {
     checkAbort(signal);
     if (!Number.isSafeInteger(reader.size) || reader.size < 0) fail("BYTE_MODE_UNSUPPORTED", "unsafe range size");
     const revision = await reader.revision();
     checkAbort(signal);
     const start = input.cursor === undefined ? 0 : decode(input.cursor as string, path, revision, reader.size);
     // One lookahead byte identifies a split codepoint; nothing larger is read.
     const length = Math.min(maxBytes + 1, reader.size - start);
     const buffer = await reader.readAt(start, length, signal);
     checkAbort(signal);
     if (await reader.revision() !== revision) cursorFailure("STALE_CURSOR", "file changed during read");
     checkAbort(signal);
     if (buffer.length !== length) cursorFailure("STALE_CURSOR", "short/inconsistent range read");
     let used = Math.min(maxBytes, buffer.length);
     if (used < buffer.length) while (used > 0 && (buffer[used] & 0xc0) === 0x80) used--;
     let text: string;
     try { text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(buffer.subarray(0, used)); }
     catch { return fail("INVALID_UTF8", "bytes mode supports valid UTF-8 only; no lossy replacement or re-encoding performed"); }
     if (!used && start < reader.size) fail("INVALID_UTF8", "no complete UTF-8 codepoint at cursor");
     const end = start + used;
     const metadata = { schema: SCHEMA, mode: "bytes", encoding: "utf-8", snapshot: hash(revision),
      startByte: start, endByte: end, returnedBytes: used, totalBytes: reader.size,
      eof: end === reader.size, nextCursor: encode(path, revision, end) };
     // EOF still supplies an end-position cursor: repeat it to observe explicit
     // zero-byte EOF (with a fresh revision check). Metadata is a separate block;
     // content[0] alone is the lossless source. Both blocks together < 50 KiB.
     return { content: [{ type: "text" as const, text }, { type: "text" as const,
      text: `[read bytes: content[0] is source only; ${metadata.eof ? "EOF" : "continue same path, mode=bytes, cursor=nextCursor"}]\n${JSON.stringify(metadata)}` }],
      details: { boundedRead: metadata }, structuredContent: metadata };
    } finally { await reader.close(); }
   }, signal);
  },
 });
}
