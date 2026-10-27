import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { runToolCall, type AgentTool } from "@earendil-works/pi-agent-core";
import { createReadTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { buildBoundedReadTool, BYTE_CHUNK_MAX, type BoundedReadOptions, type BoundedReadOperationsV1 } from "../src/harness/pi/bounded-read.js";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";
import { createToolCallBudgetExtension, type ToolCallBudget } from "../src/harness/pi/tool-call-budget.js";

// Real pinned PI adapter + real runToolCall preparation/validation/dispatch.
// No AgentSession, model/provider, SDK transport, shell, network or hardware.
const { wrapToolDefinition } = await import(new URL("./core/tools/tool-definition-wrapper.js",
 import.meta.resolve("@earendil-works/pi-coding-agent")).href) as {
 wrapToolDefinition(definition: ToolDefinition): AgentTool;
};
type Result = Awaited<ReturnType<AgentTool["execute"]>> & { isError: boolean };
type Metadata = {
 schema: string; mode: string; encoding: string; snapshot: string; startByte: number; endByte: number;
 returnedBytes: number; totalBytes: number; eof: boolean; nextCursor: string;
};
const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");
const text = (r: Result) => r.content.map(c => c.type === "text" ? c.text : "").join("\n");
const meta = (r: Result) => (r.details as { boundedRead: Metadata }).boundedRead;
function fixture(fn: (root: string) => Promise<void>) {
 return async () => {
  const root = mkdtempSync(join(tmpdir(), "rosclaw-read-continuation-"));
  try { await fn(root); } finally { rmSync(root, { recursive: true, force: true }); }
 };
}
function policyHooks(policy: ToolCallBudget) {
 const hooks: Record<string, (event: never) => Promise<unknown>> = {};
 createToolCallBudgetExtension(policy)({ on: (name: string, handler: typeof hooks[string]) => { hooks[name] = handler; } } as never);
 return hooks;
}
function runtime(root: string, options?: BoundedReadOptions, policy?: ToolCallBudget) {
 const definition = buildBoundedReadTool(root, options);
 const tool = wrapToolDefinition(definition);
 const hooks = policy && policyHooks(policy);
 let calls = 0;
 return { definition, tool, calls: () => calls,
  async call(args: Record<string, unknown>, signal?: AbortSignal): Promise<Result> {
   calls++;
   const dispatched = await runToolCall({ type: "toolCall", id: `r${calls}`, name: "read", arguments: args as never }, {
    assistantMessage: {} as never, context: { tools: [tool], messages: [] }, tools: [tool], signal,
    beforeToolCall: hooks ? async ({ args: validated }) =>
     await hooks.tool_call({ toolName: "read", input: validated } as never) as { block: true; reason: string } | undefined : undefined,
   });
   return { ...dispatched.result, isError: dispatched.isError };
  },
 };
}
async function whole(r: ReturnType<typeof runtime>, path: string, expected: Buffer, maxBytes = BYTE_CHUNK_MAX) {
 const parts: Buffer[] = [];
 let cursor: string | undefined, position = 0, snapshot: string | undefined;
 for (let i = 0; i < 10000; i++) {
  const result = await r.call({ path, mode: "bytes", maxBytes, ...(cursor ? { cursor } : {}) });
  assert.equal(result.isError, false, text(result));
  const m = meta(result);
  assert.equal(m.schema, "rosclaw.read.bytes.v1"); assert.equal(m.encoding, "utf-8");
  assert.equal(m.startByte, position); assert.equal(m.totalBytes, expected.length);
  if (snapshot) assert.equal(m.snapshot, snapshot); else snapshot = m.snapshot;
  const first = result.content[0]; assert.equal(first.type, "text");
  const body = Buffer.from(first.type === "text" ? first.text : "", "utf8");
  assert.equal(body.length, m.returnedBytes); assert.ok(body.length <= maxBytes);
  assert.equal(m.endByte, position + body.length);
  assert.deepEqual(body, expected.subarray(position, m.endByte), "each payload is exact source bytes, never a projection");
  assert.ok(Buffer.byteLength(text(result)) <= 51200, "body plus notice obey global 50 KiB cap");
  const notice = result.content[1]; assert.equal(notice.type, "text");
  assert.deepEqual(JSON.parse((notice.type === "text" ? notice.text : "").split("\n")[1]), m);
  assert.deepEqual(result.structuredContent, m);
  parts.push(body); position = m.endByte; cursor = m.nextCursor;
  if (m.eof) {
   assert.equal(position, expected.length);
   const reconstructed = Buffer.concat(parts);
   assert.deepEqual(reconstructed, expected); assert.equal(sha(reconstructed), sha(expected));
   const eof = await r.call({ path, mode: "bytes", cursor });
   assert.equal(eof.isError, false, text(eof)); assert.equal(meta(eof).eof, true);
   assert.equal(meta(eof).returnedBytes, 0); assert.equal(meta(eof).startByte, position);
   assert.equal(eof.content[0].type === "text" && eof.content[0].text, "");
   return;
  }
  assert.ok(body.length > 0, "non-EOF must progress");
 }
 assert.fail("bounded number of reads exhausted");
}
function jsonBytes(total: number, unit = "A") {
 const empty = JSON.stringify({ payload: "", tail: "TAIL_SENTINEL_END" });
 const available = total - Buffer.byteLength(empty), n = Math.floor(available / Buffer.byteLength(unit));
 const value = JSON.stringify({ payload: unit.repeat(n) + "A".repeat(available - n * Buffer.byteLength(unit)), tail: "TAIL_SENTINEL_END" });
 assert.equal(Buffer.byteLength(value), total);
 return Buffer.from(value);
}
const giant = jsonBytes(679452);
function put(root: string, name: string, bytes: Buffer | string) { writeFileSync(join(root, name), bytes); return name; }

// Thirteen historical repro cases, now invoking the SAME production factory.
test("repro 01: giant 679452B no LF: limit1 gives supported read restart and complete raw SHA", fixture(async root => {
 const path = put(root, "giant.json", giant), r = runtime(root);
 const first = await r.call({ path, offset: 1, limit: 1 });
 assert.equal(first.isError, false); assert.match(text(first), /Restart from file byte 0/); assert.doesNotMatch(text(first), /bash|sed /);
 assert.deepEqual((first.details as { continuation: { nextRead: unknown } }).continuation.nextRead, { path, mode: "bytes" });
 await whole(r, path, giant);
}));
test("repro 02: changing line limit to 1024 does not claim giant-line progress", fixture(async root => {
 const path = put(root, "giant.json", giant), r = runtime(root);
 assert.equal(text(await r.call({ path, offset: 1, limit: 1 })), text(await r.call({ path, offset: 1, limit: 1024 })));
}));
test("repro 03: no-LF offset2 retains ordinary EOF error", fixture(async root => {
 const path = put(root, "giant.json", giant), result = await runtime(root).call({ path, offset: 2, limit: 1 });
 assert.equal(result.isError, true); assert.match(text(result), /beyond end of file \(1 lines total\)/);
}));
test("repro 04: giant with LF: old offset2 empty body is not byte EOF", fixture(async root => {
 const bytes = Buffer.concat([giant, Buffer.from("\n")]), path = put(root, "lf.json", bytes), r = runtime(root);
 assert.equal(text(await r.call({ path, offset: 2, limit: 1 })), ""); await whole(r, path, bytes);
}));
test("repro 05: unknown byteOffset/byteLimit are rejected, never silently ignored", fixture(async root => {
 const path = put(root, "giant.json", giant), r = runtime(root);
 for (const extra of [{ byteOffset: 51200 }, { byteLimit: 40000 }, { byteOffset: 51200, byteLimit: 40000 }]) {
  const result = await r.call({ path, offset: 1, limit: 1, ...extra });
  assert.equal(result.isError, true); assert.match(text(result), /INVALID_READ_ARGUMENTS|Validation failed/);
 }
 await assert.rejects(r.tool.execute("direct", { path, byteOffset: 1 }), /INVALID_READ_ARGUMENTS/);
}));
test("repro 06: mixed head stays line-compatible; whole bytes retains interior giant", fixture(async root => {
 const bytes = Buffer.concat([Buffer.from("SMALL_HEAD\n"), giant, Buffer.from("\nSMALL_TAIL\n")]);
 const path = put(root, "mixed.json", bytes), r = runtime(root);
 assert.equal(text(await r.call({ path })), `SMALL_HEAD\n\n[Showing lines 1-1 of 4 (50.0KB limit). Use offset=2 to continue.]`);
 await whole(r, path, bytes);
}));
test("repro 07: interior giant line2 offers byte-zero restart, never offset3 skip", fixture(async root => {
 const bytes = Buffer.concat([Buffer.from("SMALL_HEAD\n"), giant, Buffer.from("\nSMALL_TAIL\n")]);
 const path = put(root, "mixed.json", bytes), r = runtime(root);
 const result = await r.call({ path, offset: 2, limit: 1 });
 assert.match(text(result), /Do not skip/); assert.doesNotMatch(text(result), /offset=3|bash/);
 await whole(r, path, bytes);
}));
test("repro 08: explicit line3 still means line3, with original trailing LF", fixture(async root => {
 const path = put(root, "mixed.json", Buffer.concat([Buffer.from("SMALL_HEAD\n"), giant, Buffer.from("\nSMALL_TAIL\n")]));
 assert.equal(text(await runtime(root).call({ path, offset: 3 })), "SMALL_TAIL\n");
}));
test("repro 09: giant Unicode + combining characters retains exact UTF8 bytes and SHA", fixture(async root => {
 const bytes = jsonBytes(679452, "汉🙂e\u0301"), path = put(root, "unicode.json", bytes), r = runtime(root);
 assert.match(text(await r.call({ path, offset: 1, limit: 1 })), /mode.*bytes/); await whole(r, path, bytes);
}));
for (const size of [51200, 51201]) {
 test(`repro ${size === 51200 ? "10" : "11"}: ${size} byte boundary`, fixture(async root => {
  const bytes = Buffer.alloc(size, "Q"), path = put(root, "boundary.txt", bytes), r = runtime(root);
  const line = await r.call({ path, offset: 1, limit: 1 });
  if (size === 51200) assert.deepEqual(Buffer.from(text(line)), bytes);
  else assert.match(text(line), /Restart from file byte 0/);
  await whole(r, path, bytes);
 }));
}
test("repro 12: spaces, quotes and Unicode in path need no shell escaping", fixture(async root => {
 const path = put(root, "space quote' 双\" file.json", giant), r = runtime(root);
 const first = await r.call({ path, limit: 1 });
 const next = (first.details as { continuation: { nextRead: Record<string, unknown> } }).continuation.nextRead;
 assert.equal(next.path, path); assert.equal((await r.call(next)).isError, false);
 await whole(r, path, giant);
}));
test("repro 13: normal multiline and trailing empty lines remain byte-for-byte compatible", fixture(async root => {
 const bytes = Buffer.from("one\ntwo\n\n"), path = put(root, "small.txt", bytes), r = runtime(root);
 assert.equal(text(await r.call({ path, offset: 1, limit: 1024 })), bytes.toString());
 assert.equal(text(await r.call({ path, offset: 4 })), ""); await whole(r, path, bytes);
}));

test("production wiring uses same-name read factory, schema is closed, read stays parallel", () => {
 const candidates = [new URL("../src/harness/pi/pi-runtime.ts", import.meta.url), new URL("../../src/harness/pi/pi-runtime.ts", import.meta.url)];
 const source = readFileSync(candidates.find(p => existsSync(p))!, "utf8");
 assert.match(source, /import \{ buildBoundedReadTool \} from "\.\/bounded-read.js"/);
 assert.match(source, /const customTools = filterModelTools\(\[[\s\S]*?buildBoundedReadTool\(cwd\)/);
 const r = runtime(".");
 assert.equal(r.definition.name, "read"); assert.equal(r.tool.name, "read");
 assert.notEqual(r.definition.executionMode, "sequential"); assert.notEqual(r.tool.executionMode, "sequential");
 const schema = r.definition.parameters as { additionalProperties?: boolean; properties?: Record<string, unknown> };
 assert.equal(schema.additionalProperties, false);
 assert.deepEqual(Object.keys(schema.properties ?? {}), ["path", "mode", "offset", "limit", "cursor", "maxBytes"]);
});
test("empty file and explicit EOF have unambiguous zero-byte ranges", fixture(async root => {
 const path = put(root, "empty", ""); await whole(runtime(root), path, Buffer.alloc(0));
}));
test("UTF8 cut boundaries preserve BOM, CRLF, emoji, combining marks, NUL and final blanks", fixture(async root => {
 const bytes = Buffer.from("\uFEFFa🙂汉e\u0301\r\n\u0000尾\n\n"), path = put(root, "utf8.txt", bytes), r = runtime(root);
 for (const maxBytes of [4, 5, 6, 7, 8, 11]) await whole(r, path, bytes, maxBytes);
}));
test("invalid UTF8 is a typed error, never lossy replacement or fabricated progress", fixture(async root => {
 for (const [name, bytes] of [["invalid", Buffer.from([0x61, 0x80, 0x62])], ["incomplete", Buffer.from([0xf0, 0x9f])]] as const) {
  const path = put(root, name, bytes), result = await runtime(root).call({ path, mode: "bytes" });
  assert.equal(result.isError, true); assert.match(text(result), /INVALID_UTF8/);
 }
}));
test("strict modes/parameters reject conflicts, null cursors and invalid limits before PI coercion", fixture(async root => {
 const path = put(root, "a", "abcdefgh"), r = runtime(root);
 for (const args of [
  { mode: "bytes", offset: 1 }, { mode: "bytes", limit: 1 }, { cursor: "x" }, { maxBytes: 4 },
  { mode: "lines", cursor: "x" }, { mode: "byte" }, { mode: null },
  ...[null, "4", true, 0, 3, 32769, 4.1, -1, Number.MAX_SAFE_INTEGER + 1].map(maxBytes => ({ mode: "bytes", maxBytes })),
  ...[null, "", 1, "x".repeat(2049)].map(cursor => ({ mode: "bytes", cursor })),
 ]) {
  const result = await r.call({ path, ...args });
  assert.equal(result.isError, true, JSON.stringify(args)); assert.match(text(result), /READ_MODE_CONFLICT|INVALID_READ_ARGUMENTS/);
 }
 const direct = r.tool.execute("direct", { path, mode: "bytes", cursor: null });
 await assert.rejects(direct, /INVALID_READ_ARGUMENTS/);
}));
test("malformed, forged/beyond-EOF, wrong-path and wrong-instance cursors are rejected", fixture(async root => {
 const path = put(root, "a", "abcdefghijkl"), other = put(root, "b", "abcdefghijkl"), r = runtime(root);
 const first = await r.call({ path, mode: "bytes", maxBytes: 4 }), cursor = meta(first).nextCursor;
 const data = JSON.parse(Buffer.from(cursor.split(".")[0], "base64url").toString()); data[3] = 999999;
 const beyond = Buffer.from(JSON.stringify(data)).toString("base64url") + "." + cursor.split(".")[1];
 for (const bad of ["bad", "x.y.z", "a.b", beyond, "!" + cursor]) {
  const result = await r.call({ path, mode: "bytes", cursor: bad });
  assert.equal(result.isError, true); assert.match(text(result), /INVALID_CURSOR/);
 }
 assert.match(text(await r.call({ path: other, mode: "bytes", cursor })), /INVALID_CURSOR/);
 assert.match(text(await runtime(root).call({ path, mode: "bytes", cursor })), /INVALID_CURSOR/);
}));
test("same-size pathname replacement rejects old cursor; fresh restart reconstructs replacement", fixture(async root => {
 const path = put(root, "a", "abcdefghijklmno"), r = runtime(root);
 const cursor = meta(await r.call({ path, mode: "bytes", maxBytes: 4 })).nextCursor;
 const replacement = "ABCDEFGHIJKLMNO"; put(root, "new", replacement); renameSync(join(root, "new"), join(root, path));
 const stale = await r.call({ path, mode: "bytes", cursor });
 assert.equal(stale.isError, true); assert.match(text(stale), /STALE_CURSOR/);
 await whole(r, path, Buffer.from(replacement));
}));
test("actual guarded edit interleaved between reads invalidates cursor, no cache bypass", fixture(async root => {
 const path = put(root, "a", "abcdefghijklmno"), r = runtime(root);
 const cursor = meta(await r.call({ path, mode: "bytes", maxBytes: 4 })).nextCursor;
 const edit = wrapToolDefinition(buildWorkspacePackTools({ root }).find(t => t.name === "edit")!);
 const changed = await runToolCall({ type: "toolCall", id: "edit", name: "edit", arguments: { path, oldText: "efgh", newText: "EFGH" } }, {
  assistantMessage: {} as never, context: { tools: [edit], messages: [] }, tools: [edit],
 });
 assert.equal(changed.isError, false);
 for (let i = 0; i < 2; i++) {
  const stale = await r.call({ path, mode: "bytes", cursor }); assert.equal(stale.isError, true); assert.match(text(stale), /STALE_CURSOR/);
 }
 await whole(r, path, Buffer.from("abcdEFGHijklmno"));
}));
test("EOF cursor also rechecks file changes rather than returning cached EOF", fixture(async root => {
 const path = put(root, "a", "small"), r = runtime(root);
 const first = await r.call({ path, mode: "bytes" }); assert.equal(meta(first).eof, true);
 writeFileSync(join(root, path), "SMALL");
 assert.match(text(await r.call({ path, mode: "bytes", cursor: meta(first).nextCursor })), /STALE_CURSOR/);
}));

function gate() {
 let open!: () => void;
 const promise = new Promise<void>(resolve => { open = resolve; });
 return { promise, open };
}
function memoryRanges(bytes: Buffer, onRead?: () => Promise<void>) {
 const state = { revision: 0, opens: 0, closes: 0, requests: [] as number[] };
 const operations: BoundedReadOperationsV1 = { version: 1, async open() {
  state.opens++;
  return { size: bytes.length, revision: async () => String(state.revision),
   async readAt(offset, length) { state.requests.push(length); await onRead?.(); return bytes.subarray(offset, offset + length); },
   async close() { state.closes++; },
  };
 } };
 return { state, operations };
}
test("range IO reads only requested bytes plus one lookahead, not the entire giant file", fixture(async root => {
 const seam = memoryRanges(giant), r = runtime(root, { boundedOperations: seam.operations });
 const first = await r.call({ path: "remote", mode: "bytes", maxBytes: 4 });
 assert.equal(first.isError, false); assert.deepEqual(seam.state.requests, [5]);
 const second = await r.call({ path: "remote", mode: "bytes", maxBytes: 7, cursor: meta(first).nextCursor });
 assert.equal(second.isError, false); assert.deepEqual(seam.state.requests, [5, 8]);
 assert.equal(seam.state.opens, 2); assert.equal(seam.state.closes, 2);
}));
test("mutation during a range read discards payload and always closes", fixture(async root => {
 const seam = memoryRanges(Buffer.from("abcdefgh"), async () => { seam.state.revision++; });
 const result = await runtime(root, { boundedOperations: seam.operations }).call({ path: "remote", mode: "bytes", maxBytes: 4 });
 assert.equal(result.isError, true); assert.match(text(result), /STALE_CURSOR/); assert.equal(seam.state.closes, 1);
 assert.equal((result.details as { boundedRead?: unknown })?.boundedRead, undefined);
}));
test("short remote read is rejected instead of fabricating EOF", fixture(async root => {
 let closed = 0;
 const operations: BoundedReadOperationsV1 = { version: 1, async open() { return {
  size: 100, revision: async () => "1", readAt: async () => Buffer.from("x"), close: async () => { closed++; },
 }; } };
 const result = await runtime(root, { boundedOperations: operations }).call({ path: "remote", mode: "bytes" });
 assert.equal(result.isError, true); assert.match(text(result), /STALE_CURSOR/); assert.equal(closed, 1);
}));
test("parallel bytes reads overlap and issue independent consistent cursors", { timeout: 10000 }, fixture(async root => {
 const entered = gate(), release = gate(); let active = 0, peak = 0;
 const bytes = Buffer.from("abcdefghijkl"), seam = memoryRanges(bytes, async () => {
  active++; peak = Math.max(peak, active); if (active === 2) entered.open(); await release.promise; active--;
 });
 const r = runtime(root, { boundedOperations: seam.operations });
 const a = r.call({ path: "remote", mode: "bytes", maxBytes: 4 });
 const b = r.call({ path: "remote", mode: "bytes", maxBytes: 7 });
 try {
  await entered.promise; assert.equal(active, 2); assert.equal(peak, 2); release.open();
  const [ra, rb] = await Promise.all([a, b]); assert.equal(ra.isError, false); assert.equal(rb.isError, false);
  assert.equal(meta(ra).endByte, 4); assert.equal(meta(rb).endByte, 7); assert.equal(seam.state.closes, 2);
 } finally { release.open(); }
}));
test("in-flight bytes abort settles promptly, closes late IO, and leaves next call unlocked", { timeout: 10000 }, fixture(async root => {
 const entered = gate(), release = gate(), closed = gate(); let calls = 0;
 const seam = memoryRanges(Buffer.from("abcdefghijkl"), async () => { if (++calls === 1) { entered.open(); await release.promise; } });
 const originalOpen = seam.operations.open;
 seam.operations.open = async (...args) => {
  const reader = await originalOpen(...args), close = reader.close;
  reader.close = async () => { await close(); if (seam.state.closes === 2) closed.open(); }; return reader;
 };
 const r = runtime(root, { boundedOperations: seam.operations }), controller = new AbortController();
 const pending = r.call({ path: "remote", mode: "bytes", maxBytes: 4 }, controller.signal);
 try {
  await entered.promise; controller.abort(); const aborted = await pending;
  assert.equal(aborted.isError, true); assert.equal(text(aborted), "Operation aborted");
  const next = await r.call({ path: "remote", mode: "bytes", maxBytes: 4 });
  assert.equal(next.isError, false); assert.equal(meta(next).endByte, 4);
  release.open(); await closed.promise;
 } finally { release.open(); }
}));
test("pre-aborted bytes read performs no remote IO", fixture(async root => {
 const seam = memoryRanges(giant), r = runtime(root, { boundedOperations: seam.operations }), controller = new AbortController(); controller.abort();
 const result = await r.call({ path: "remote", mode: "bytes" }, controller.signal);
 assert.equal(result.isError, true); assert.equal(text(result), "Operation aborted"); assert.equal(seam.state.opens, 0);
}));
test("legacy remote seam stays remote: line compatibility/full-file IO, byte mode explicit block", fixture(async root => {
 let reads = 0;
 const options = { operations: { access: async () => undefined, detectImageMimeType: async () => undefined,
  readFile: async () => { reads++; return giant; } } };
 const r = runtime(root, options);
 const line = await r.call({ path: "remote", limit: 1 });
 assert.equal(reads, 1); assert.match(text(line), /BYTE_MODE_UNSUPPORTED/); assert.doesNotMatch(text(line), /bash/);
 const blocked = await r.call({ path: "remote", mode: "bytes" });
 assert.equal(blocked.isError, true); assert.match(text(blocked), /BYTE_MODE_UNSUPPORTED/); assert.equal(reads, 1);
 const seam = memoryRanges(giant), enabled = runtime(root, { ...options, boundedOperations: seam.operations });
 await whole(enabled, "remote", giant); assert.equal(reads, 1);
}));
test("ordinary line results, errors, odd Number semantics and structuredContent match upstream", fixture(async root => {
 const bytes = Buffer.from("one\r\ntwo\n三🙂\n\n"), path = put(root, "small.txt", bytes), r = runtime(root), upstream = createReadTool(root);
 for (const args of [{ path }, { path, offset: 2, limit: 1 }, { path, offset: 0 }, { path, offset: -2, limit: 2 },
  { path, offset: 1.5, limit: 2.5 }, { path, limit: 0 }, { path, offset: 4 }, { path, offset: 5 }]) {
  const actual = await r.call(args), expected = await upstream.execute("upstream", args);
  assert.equal(actual.isError, false); assert.deepEqual(actual.content, expected.content);
  assert.deepEqual(actual.details, expected.details); assert.deepEqual(actual.structuredContent, expected.structuredContent);
  const explicit = await r.call({ ...args, mode: "lines" }); assert.deepEqual(explicit.content, expected.content);
 }
 for (const args of [{ path, offset: 99 }, { path: "missing" }]) {
  const actual = await r.call(args); assert.equal(actual.isError, true);
  await assert.rejects(upstream.execute("upstream", args), error => { assert.equal(text(actual), (error as Error).message); return true; });
 }
}));
test("default image behavior delegates to installed PI unchanged, including model note", fixture(async root => {
 const image = Buffer.from("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", "base64");
 const path = put(root, "pixel.gif", image), r = runtime(root), upstream = createReadTool(root);
 const actual = await r.call({ path }), expected = await upstream.execute("upstream", { path });
 assert.equal(actual.isError, false); assert.deepEqual(actual.content, expected.content);
 assert.deepEqual(actual.structuredContent, expected.structuredContent); assert.match(text(actual), /Read image file/);
 // Context forwarding uses the actual definition with non-vision model metadata.
 const ctx = { cwd: root, model: { input: ["text"] } } as never;
 const delegated = await r.definition.execute("ctx", { path }, undefined, undefined, ctx);
 assert.match(JSON.stringify(delegated.content), /does not support images/);
}));
for (const exact of [false, true]) {
 test(`giant completes under ${exact ? "exact-selftest Bash" : "read-only"} policy with no Bash calls`, fixture(async root => {
  const policy: ToolCallBudget = exact
   ? { allowedTools: ["read", "edit", "bash"], maxCalls: { read: 64, edit: 1, bash: 1 }, maxTotalCalls: 66, exactCommands: { bash: ["ONLY_PREAUTHORIZED_SELFTEST_COMMAND"] }, visibleBudget: true }
   : { allowedTools: ["read"], maxCalls: { read: 64 }, maxTotalCalls: 64, visibleBudget: true };
  const path = put(root, "giant.json", giant), r = runtime(root, undefined, policy);
  await whole(r, path, giant); assert.ok(r.calls() > 1);
  const hooks = policyHooks(policy), denied = await hooks.tool_call({ toolName: "bash", input: { command: "sed giant.json" } } as never) as { block: boolean; reason: string };
  assert.equal(denied.block, true); assert.match(denied.reason, exact ? /EXACT_COMMAND_REJECTED/ : /TOOL_NOT_ALLOWED/);
 }));
}
