import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createReadTool } from "@earendil-works/pi-coding-agent";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";
import { Agent, type AgentEvent, type AgentTool, type AgentToolCall, type StreamFn } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, AssistantMessageEventStream, Model, ToolResultMessage } from "@earendil-works/pi-ai";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";

// Exercise the installed definition-to-runtime adapter, not a copy of it.
// Resolve beside PI's package entry so this also works from dist/test; no
// extension loading, model runtime, provider registration or credentials.
const wrapperUrl = new URL("./core/tools/tool-definition-wrapper.js",
 import.meta.resolve("@earendil-works/pi-coding-agent"));
const { wrapToolDefinition } = await import(wrapperUrl.href) as {
 wrapToolDefinition: (definition: ToolDefinition) => AgentTool;
};

const offlineModel: Model<"openai-responses"> = {
 id: "offline-order-fixture", name: "offline-order-fixture", api: "openai-responses",
 provider: "offline-fixture", baseUrl: "", reasoning: false, input: ["text"],
 cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}, contextWindow: 100000, maxTokens: 4096,
};
function scriptedStream(batches: AgentToolCall[][]): {streamFn: StreamFn; requests: () => number} {
 let requests = 0;
 const streamFn: StreamFn = () => {
  const calls = batches[requests++];
  const message: AssistantMessage = {
   role: "assistant", content: calls ?? [{type: "text", text: "fixture settled"}],
   api: offlineModel.api, provider: offlineModel.provider, model: offlineModel.id,
   usage: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0}},
   stopReason: calls ? "toolUse" : "stop", timestamp: 0,
  };
  // Only the public stream consumption protocol is faked. Dispatch, validation,
  // hooks, cancellation, tool results and Agent lifecycle are the real PI loop.
  return {
   async *[Symbol.asyncIterator]() {
    yield {type: "start", partial: message};
    yield {type: "done", reason: message.stopReason, message};
   },
   result: async () => message,
  } as unknown as AssistantMessageEventStream;
 };
 return {streamFn, requests: () => requests};
}
function toolCall(id: string, name: string, args: AgentToolCall["arguments"]): AgentToolCall {
 return {type: "toolCall", id, name, arguments: args};
}
function dispatchFixture(tools: AgentTool[], batches: AgentToolCall[][], options: {
 block?: string; abortBefore?: string; afterError?: string;
} = {}) {
 const log: string[] = [], events: AgentEvent[] = [];
 const script = scriptedStream(batches);
 const instrumented = tools.map(tool => ({...tool, execute: async (...args: Parameters<AgentTool["execute"]>) => {
  log.push(`execute:${args[0]}`);
  try {
   const result = await tool.execute(...args);
   log.push(`return:${args[0]}`);
   return result;
  } catch (error) { log.push(`throw:${args[0]}`); throw error; }
 }}));
 const agent = new Agent({
  initialState: {model: offlineModel, tools: instrumented}, streamFn: script.streamFn,
  // Deliberately leave toolExecution at PI's parallel default. The product
  // definitions, carried through its real wrapper, must select the barrier.
  beforeToolCall: async ({toolCall}) => {
   log.push(`before:${toolCall.id}`);
   if (toolCall.id === options.abortBefore) agent.abort();
   if (toolCall.id === options.block) return {block: true, reason: "fixture budget denied"};
   return undefined;
  },
  afterToolCall: async ({toolCall}) => {
   log.push(`after:${toolCall.id}`);
   if (toolCall.id === options.afterError) throw new Error("fixture finalize exception");
   return undefined;
  },
  finishTurn: (_turn, signal) => signal?.aborted ? {action: "end"} : undefined,
 });
 agent.subscribe(event => {
  events.push(event);
  if (event.type === "tool_execution_start") log.push(`start:${event.toolCallId}`);
  if (event.type === "tool_execution_end") log.push(`end:${event.toolCallId}`);
  if (event.type === "message_end" && event.message.role === "toolResult") log.push(`ack:${event.message.toolCallId}`);
 });
 return {agent, log, events, requests: script.requests,
  run: async () => { await agent.prompt("offline scripted tool batch"); await agent.waitForIdle(); },
  results: () => agent.state.messages.filter((m): m is ToolResultMessage => m.role === "toolResult"),
 };
}
function packRuntime(root: string, beforeEffect?: () => Promise<void>) {
 const definitions = buildWorkspacePackTools({root, bwrapPath: () => null, beforeEffect});
 const tools = [createReadTool(root), ...definitions.map(d => wrapToolDefinition(d))];
 for (const name of ["edit", "write", "bash"]) {
  assert.equal(definitions.find(d => d.name === name)!.executionMode, "sequential");
  assert.equal(tools.find(t => t.name === name)!.executionMode, "sequential");
 }
 assert.notEqual(tools[0].executionMode, "sequential");
 return tools;
}
function resultText(result: ToolResultMessage): string {
 return result.content.map(c => c.type === "text" ? c.text : "").join("\n");
}
function assertSerial(log: string[], ids: string[]) {
 assert.deepEqual(log, ids.flatMap(id => [
  `start:${id}`, `before:${id}`, `execute:${id}`, `return:${id}`,
  `after:${id}`, `end:${id}`, `ack:${id}`,
 ]));
}
function assertSettled(h: ReturnType<typeof dispatchFixture>) {
 assert.equal(h.agent.state.isStreaming, false);
 assert.equal(h.agent.state.pendingToolCalls.size, 0);
 assert.equal(h.events.filter(e => e.type === "agent_end").length, 1);
 assert.equal(h.agent.state.errorMessage, undefined);
}

// Synthetic IO boundary only; no claim about historical author16147/U1.
// Grow an interior token by exactly 17 ASCII bytes; unchanged EOF has two LFs.
const boundaryOld = Array.from({length: 1172}, (_, i) =>
 i === 735 ? "EDIT_ANCHOR" : `LINE_${String(i + 1).padStart(4, "0")}_${"x".repeat(50)}`
).join("\n") + "\nTAIL_0123456789\n\n";
const boundaryNew = boundaryOld.replace("EDIT_ANCHOR", "EDIT_ANCHOR" + "G".repeat(17));
const boundaryRead = (id: string, offset = 735, limit = 1024) => toolCall(id, "read", {path: "a.txt", offset, limit});
const boundaryEdit = (id: string, oldText = "EDIT_ANCHOR", newText = "EDIT_ANCHOR" + "G".repeat(17)) =>
 toolCall(id, "edit", {path: "a.txt", oldText, newText});
const fromLine = (text: string, offset: number) => text.split("\n").slice(offset - 1).join("\n");

for (const writer of ["edit", "write", "bash"] as const) {
 for (const readFirst of [true, false]) {
  test(`real PI batch: ${readFirst ? "READ before" : "READ after"} ${writer} observes complete ordered bytes`, async () => {
   const root = mkdtempSync(join(tmpdir(), "filetool-pi-order-"));
   try {
    assert.equal(Buffer.byteLength(boundaryNew) - Buffer.byteLength(boundaryOld), 17);
    assert.ok(Buffer.byteLength(boundaryOld) > 64 * 1024, "exercise multi-chunk input, not output truncation");
    assert.equal(boundaryNew.slice(-17), "TAIL_0123456789\n\n");
    writeFileSync(join(root, "a.txt"), boundaryOld);
    writeFileSync(join(root, "grown.txt"), boundaryNew);
    let effects = 0;
    const tools = packRuntime(root, async () => { effects++; });
    const write = writer === "edit" ? boundaryEdit("w") : writer === "write"
     ? toolCall("w", "write", {path: "a.txt", content: boundaryNew})
     // Real guarded foreground bash: admission/env/cwd/close-result behavior is
     // unchanged. No background job, model/provider request or network.
     : toolCall("w", "bash", {command: "cp grown.txt a.txt"});
    const calls = readFirst ? [boundaryRead("r"), write] : [write, boundaryRead("r")];
    const h = dispatchFixture(tools, [calls]);
    await h.run();
    assertSerial(h.log, calls.map(c => c.id));
    const results = h.results();
    assert.deepEqual(results.map(r => r.toolCallId), calls.map(c => c.id));
    assert.ok(results.every(r => !r.isError));
    const read = results.find(r => r.toolName === "read")!;
    assert.equal(resultText(read), fromLine(readFirst ? boundaryOld : boundaryNew, 735));
    assert.ok(resultText(read).endsWith("\n\n"), "both trailing empty lines survive");
    assert.doesNotMatch(resultText(read), /more lines|Showing lines|TRUNCATED/);
    assert.equal(readFileSync(join(root, "a.txt"), "utf8"), boundaryNew);
    const written = results.find(r => r.toolCallId === "w")!;
    if (writer === "edit") assert.equal(resultText(written), `edited ${join(root, "a.txt")}`);
    if (writer === "write") assert.equal(resultText(written), `wrote ${join(root, "a.txt")} (${Buffer.byteLength(boundaryNew)} bytes)`);
    if (writer === "bash") {
     const details = written.details as Record<string, unknown>;
     assert.equal(details.exitCode, 0); assert.equal(details.signal, null);
     assert.equal(details.timedOut, false); assert.equal(details.aborted, false);
     assert.equal(details.truncated, false); assert.equal(details.outputLimitBytes, 65536);
     assert.match(resultText(written), /TOOL_LAYER_ONLY/);
    }
    assert.equal(effects, 1); assert.equal(h.requests(), 2);
    assertSettled(h);
   } finally { rmSync(root, {recursive: true, force: true}); }
  });
 }
}

for (const readFirst of [true, false]) {
 test(`real PI reconstructs ALL ${readFirst ? "pre-edit" : "post-edit"} bytes through bounded public reads`, async () => {
  const root = mkdtempSync(join(tmpdir(), "filetool-pi-complete-"));
  try {
   writeFileSync(join(root, "a.txt"), boundaryOld);
   const reads = [boundaryRead("head", 1, 734), boundaryRead("tail", 735)];
   const calls = readFirst ? [...reads, boundaryEdit("edit")] : [boundaryEdit("edit"), ...reads];
   const h = dispatchFixture(packRuntime(root), [calls]);
   await h.run();
   assertSerial(h.log, calls.map(c => c.id));
   const results = h.results().filter(r => r.toolName === "read");
   assert.ok(h.results().every(r => !r.isError));
   const expected = readFirst ? boundaryOld : boundaryNew;
   const remaining = expected.split("\n").length - 734;
   const notice = `\n\n[${remaining} more lines in file. Use offset=735 to continue.]`;
   const head = resultText(results[0]);
   assert.ok(head.endsWith(notice), "the original public continuation notice must survive");
   const reconstructed = head.slice(0, -notice.length) + "\n" + resultText(results[1]);
   assert.deepEqual(Buffer.from(reconstructed), Buffer.from(expected));
   assert.equal(reconstructed.split("\n").length, expected.split("\n").length);
   assert.ok(reconstructed.endsWith("\n\n"));
   assert.equal(readFileSync(join(root, "a.txt"), "utf8"), boundaryNew);
   assertSettled(h);
  } finally { rmSync(root, {recursive: true, force: true}); }
 });
}

test("real PI repeated edits finalize and ACK each result before preparing the next call", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-pi-repeat-"));
 try {
  writeFileSync(join(root, "a.txt"), boundaryOld);
  const state1 = boundaryNew;
  const state2 = state1.replace("G".repeat(17), "汉字" + "H".repeat(11)); // still 17 UTF8 bytes
  const state3 = state2.replace("汉字" + "H".repeat(11), "I".repeat(17));
  let effects = 0;
  const h = dispatchFixture(packRuntime(root, async () => { effects++; }), [[
   boundaryRead("r0"), boundaryEdit("e1"), boundaryRead("r1"),
   boundaryEdit("e2", "G".repeat(17), "汉字" + "H".repeat(11)), boundaryRead("r2"),
  ], [boundaryEdit("e3", "汉字" + "H".repeat(11), "I".repeat(17)), boundaryRead("r3")]]);
  const ackBodies: string[] = [];
  h.agent.subscribe(event => {
   if (event.type === "message_end" && event.message.role === "toolResult"
    && event.message.toolName === "edit") ackBodies.push(readFileSync(join(root, "a.txt"), "utf8"));
  });
  await h.run();
  assertSerial(h.log, ["r0", "e1", "r1", "e2", "r2", "e3", "r3"]);
  assert.deepEqual(ackBodies, [state1, state2, state3]);
  const results = h.results();
  assert.deepEqual(results.filter(r => r.toolName === "read").map(resultText),
   [boundaryOld, state1, state2, state3].map(s => fromLine(s, 735)));
  assert.ok(results.every(r => !r.isError));
  for (const r of results.filter(r => r.toolName === "edit")) assert.equal(resultText(r), `edited ${join(root, "a.txt")}`);
  assert.equal(effects, 3); assert.equal(h.requests(), 3);
  assert.equal(readFileSync(join(root, "a.txt"), "utf8"), state3);
  assertSettled(h);
 } finally { rmSync(root, {recursive: true, force: true}); }
});

function asyncGate() {
 let open!: () => void;
 const promise = new Promise<void>(resolve => { open = resolve; });
 return {promise, open};
}

test("real PI read-only batch overlaps at an async IO gate with writers registered but absent", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-pi-overlap-"));
 const release = asyncGate(), bothEntered = asyncGate();
 let entered = 0, active = 0, maxActive = 0;
 try {
  writeFileSync(join(root, "a.txt"), "read-only\n\n");
  const read = createReadTool(root, {operations: {
   access: async () => undefined,
   detectImageMimeType: async () => undefined,
   readFile: async () => {
    entered++; active++; maxActive = Math.max(maxActive, active);
    if (entered === 2) bothEntered.open();
    try { await release.promise; return Buffer.from("read-only\n\n"); }
    finally { active--; }
   },
  }});
  let effects = 0;
  const h = dispatchFixture([read, ...packRuntime(root, async () => { effects++; }).filter(t => t.name !== "read")], [[
   boundaryRead("r1", 1), boundaryRead("r2", 1),
  ]]);
  const pending = h.run();
  await bothEntered.promise;
  assert.equal(active, 2); assert.equal(maxActive, 2);
  assert.equal(h.results().length, 0, "neither read can settle before the gate opens");
  assert.ok(!h.log.some(s => s.startsWith("return:")));
  release.open(); await pending;
  assert.deepEqual(h.results().map(r => r.toolCallId), ["r1", "r2"]);
  assert.deepEqual(h.results().map(resultText), ["read-only\n\n", "read-only\n\n"]);
  assert.equal(entered, 2); assert.equal(active, 0); assert.equal(effects, 0);
  assertSettled(h);
 } finally { release.open(); rmSync(root, {recursive: true, force: true}); }
});

test("real PI mixed batch retains path, exact-match, IO and budget denials without poisoning later calls", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-pi-denial-"));
 const outside = mkdtempSync(join(tmpdir(), "filetool-pi-outside-"));
 try {
  writeFileSync(join(root, "a.txt"), "duplicate duplicate\n\n");
  writeFileSync(join(outside, "a.txt"), "outside"); symlinkSync(outside, join(root, "escape"));
  let effects = 0;
  const h = dispatchFixture(packRuntime(root, async () => { effects++; }), [[
   toolCall("scope", "write", {path: "../outside.txt", content: "bad"}),
   toolCall("link", "edit", {path: "escape/a.txt", oldText: "outside", newText: "bad"}),
   boundaryEdit("empty", "", "bad"), boundaryEdit("missing", "absent", "bad"), boundaryEdit("multiple", "duplicate", "bad"),
   toolCall("io", "edit", {path: "no-such-file", oldText: "x", newText: "bad"}),
   toolCall("budget", "write", {path: "a.txt", content: "budget bypass"}),
   boundaryRead("r", 1), toolCall("good", "write", {path: "u.txt", content: "汉字"}),
  ]], {block: "budget"});
  await h.run();
  const results = h.results();
  assert.deepEqual(results.slice(0, 6).map(r => (r.details as Record<string, unknown>).code), [
   "PATH_SCOPE_DENIED", "PATH_SCOPE_DENIED", "EDIT_OLD_TEXT_EMPTY", "EDIT_OLD_TEXT_NOT_FOUND", "EDIT_OLD_TEXT_NOT_UNIQUE", "FILE_IO_ERROR",
  ]);
  assert.ok(results.slice(0, 7).every(r => r.isError));
  assert.equal(resultText(results[6]), "fixture budget denied");
  assert.equal(resultText(results[7]), "duplicate duplicate\n\n");
  assert.match(resultText(results[8]), /\(6 bytes\)/); assert.equal(results[8].isError, false);
  assert.equal(effects, 7, "denied preflight must not execute or consume effect admission");
  assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "duplicate duplicate\n\n");
  assert.equal(readFileSync(join(outside, "a.txt"), "utf8"), "outside");
  assert.equal(readFileSync(join(root, "u.txt"), "utf8"), "汉字");
  const expected = ["scope", "link", "empty", "missing", "multiple", "io", "budget", "r", "good"].flatMap(id =>
   id === "budget" ? [`start:${id}`, `before:${id}`, `end:${id}`, `ack:${id}`] : [
    `start:${id}`, `before:${id}`, `execute:${id}`, `return:${id}`, `after:${id}`, `end:${id}`, `ack:${id}`,
   ]);
  assert.deepEqual(h.log, expected);
  assertSettled(h);
 } finally { rmSync(root, {recursive: true, force: true}); rmSync(outside, {recursive: true, force: true}); }
});

for (const writer of ["edit", "write", "bash"] as const) {
 test(`real PI ${writer} admission exception propagates unchanged, then READ and next effect settle`, async () => {
  const root = mkdtempSync(join(tmpdir(), "filetool-pi-admission-"));
  try {
   writeFileSync(join(root, "a.txt"), "original\n\n");
   let effects = 0;
   const tools = packRuntime(root, async () => {
    if (++effects === 1) throw new Error("REJECTED [FIXTURE_ADMISSION_DENIED]: original reason");
   });
   const bad = writer === "edit" ? boundaryEdit("bad", "original", "mutated") : writer === "write"
    ? toolCall("bad", "write", {path: "a.txt", content: "mutated"})
    : toolCall("bad", "bash", {command: "printf mutated > a.txt"});
   const h = dispatchFixture(tools, [[bad, boundaryRead("r", 1), toolCall("good", "write", {path: "u.txt", content: "allowed"})]]);
   await h.run();
   const results = h.results();
   assert.equal(results[0].isError, true);
   assert.equal(resultText(results[0]), "REJECTED [FIXTURE_ADMISSION_DENIED]: original reason");
   assert.equal(resultText(results[1]), "original\n\n"); assert.equal(results[1].isError, false);
   assert.equal(results[2].isError, false); assert.equal(effects, 2);
   assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "original\n\n");
   assertSerial(h.log.map(s => s === "throw:bad" ? "return:bad" : s), ["bad", "r", "good"]);
   assert.ok(h.log.includes("throw:bad")); assertSettled(h);
  } finally { rmSync(root, {recursive: true, force: true}); }
 });
}

test("real PI read exception is not suppressed or retried; finalize exception is preserved", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-pi-exception-"));
 try {
  writeFileSync(join(root, "a.txt"), "old");
  const h = dispatchFixture(packRuntime(root), [[
   toolCall("missing", "read", {path: "no-such-file"}), boundaryEdit("edit", "old", "new"), boundaryRead("r", 1),
  ]], {afterError: "edit"});
  await h.run();
  const results = h.results();
  assert.equal(results[0].isError, true); assert.match(resultText(results[0]), /ENOENT/);
  assert.equal(results[1].isError, true); assert.equal(resultText(results[1]), "fixture finalize exception");
  assert.equal(resultText(results[2]), "new"); assert.equal(results[2].isError, false);
  assert.equal(h.log.filter(s => s === "execute:missing").length, 1);
  assert.equal(h.log.filter(s => s === "execute:edit").length, 1);
  assert.ok(h.log.indexOf("ack:missing") < h.log.indexOf("before:edit"));
  assert.ok(h.log.indexOf("ack:edit") < h.log.indexOf("before:r"));
  assertSettled(h);
 } finally { rmSync(root, {recursive: true, force: true}); }
});

test("real PI in-flight read cancellation propagates through the wrapper and a later run is not locked", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-pi-read-abort-"));
 const entered = asyncGate(), release = asyncGate();
 let reads = 0, effects = 0;
 try {
  writeFileSync(join(root, "a.txt"), "old\n\n");
  const read = createReadTool(root, {operations: {
   access: async () => undefined,
   detectImageMimeType: async () => undefined,
   readFile: async () => {
    reads++; entered.open(); await release.promise;
    return Buffer.from(readFileSync(join(root, "a.txt")));
   },
  }});
  const h = dispatchFixture([read, ...packRuntime(root, async () => { effects++; }).filter(t => t.name !== "read")], [[
   boundaryRead("cancel-read", 1), boundaryEdit("never", "old", "bad"),
  ], [boundaryEdit("next-edit", "old", "new"), boundaryRead("next-read", 1)]]);
  const pending = h.run();
  await entered.promise;
  assert.equal(reads, 1); assert.equal(h.agent.state.pendingToolCalls.size, 1);
  h.agent.abort(); release.open(); await pending;
  assert.deepEqual(h.results().map(r => r.toolCallId), ["cancel-read"]);
  assert.equal(h.results()[0].isError, true);
  assert.equal(resultText(h.results()[0]), "Operation aborted");
  assert.ok(h.log.includes("throw:cancel-read")); assert.ok(!h.log.includes("start:never"));
  assert.equal(effects, 0); assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "old\n\n");
  assertSettled(h);
  // A new Agent run supplies a fresh signal. No barrier queue/lease from the
  // cancelled batch may survive and block it, nor may the cancelled IO ACK late.
  await h.run();
  assert.deepEqual(h.results().map(r => r.toolCallId), ["cancel-read", "next-edit", "next-read"]);
  assert.ok(h.results().slice(1).every(r => !r.isError));
  assert.equal(resultText(h.results()[2]), "new\n\n");
  assert.equal(reads, 2); assert.equal(effects, 1); assert.equal(h.requests(), 3);
  assert.equal(h.events.filter(e => e.type === "agent_end").length, 2);
  assert.equal(h.agent.state.isStreaming, false); assert.equal(h.agent.state.pendingToolCalls.size, 0);
 } finally { release.open(); rmSync(root, {recursive: true, force: true}); }
});

test("real PI abort during writer preflight settles once without executing it or later calls", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-pi-abort-"));
 try {
  writeFileSync(join(root, "a.txt"), "old\n\n");
  let effects = 0;
  const h = dispatchFixture(packRuntime(root, async () => { effects++; }), [[
   boundaryRead("r", 1), boundaryEdit("cancel", "old", "bad"), toolCall("never", "write", {path: "a.txt", content: "bad"}),
  ]], {abortBefore: "cancel"});
  await h.run();
  assert.deepEqual(h.results().map(r => r.toolCallId), ["r", "cancel"]);
  assert.equal(resultText(h.results()[0]), "old\n\n");
  assert.equal(h.results()[1].isError, true); assert.equal(resultText(h.results()[1]), "Operation aborted");
  assert.equal(effects, 0); assert.equal(h.requests(), 1);
  assert.ok(!h.log.includes("execute:cancel")); assert.ok(!h.log.includes("start:never"));
  assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "old\n\n");
  assertSettled(h);
 } finally { rmSync(root, {recursive: true, force: true}); }
});


test("absolute PI read then relative edit diagnoses workspace root; shell cd does not rebind", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-diagnostic-"));
 try {
  const root = join(base, "default"), stage = join(base, "stage"); mkdirSync(root); mkdirSync(stage);
  writeFileSync(join(root, "driver.py"), "default body"); writeFileSync(join(stage, "driver.py"), "stage body");
  assert.match(JSON.stringify((await createReadTool(root).execute("read", {path: join(stage, "driver.py")})).content), /stage body/);
  const tools = buildWorkspacePackTools({root, mode: () => "SIMULATION", bwrapPath: () => null});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  await call("bash", {command: `cd '${stage}' && pwd`});
  const relative = await call("edit", {path: "driver.py", oldText: "stage body", newText: "replacement"});
  assert.equal(relative.isError, true); assert.equal((relative.details as Record<string, unknown>).code, "EDIT_OLD_TEXT_NOT_FOUND");
  assert.equal((relative.details as Record<string, unknown>).workspace_root, root); assert.equal((relative.details as Record<string, unknown>).resolved_target, join(root, "driver.py"));
  assert.match(JSON.stringify(relative.content), /shell.*cd.*does not change/i);
  const outside = await call("edit", {path: join(stage, "driver.py"), oldText: "stage body", newText: "replacement"});
  assert.equal((outside.details as Record<string, unknown>).code, "PATH_SCOPE_DENIED");
  assert.equal(readFileSync(join(stage, "driver.py"), "utf8"), "stage body"); assert.equal(readFileSync(join(root, "driver.py"), "utf8"), "default body");
 } finally {rmSync(base, {recursive: true, force: true});}
});

test("extra-root edits retain semantics and symlink escape remains denied", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-root-")), scratch = mkdtempSync(join(tmpdir(), "filetool-scratch-"));
 try {
  writeFileSync(join(scratch, "driver.py"), "duplicate duplicate"); symlinkSync(scratch, join(root, "link"));
  const tools = buildWorkspacePackTools({root, extraRoots: () => [scratch]});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  const multiple = await call("edit", {path: join(scratch, "driver.py"), oldText: "duplicate", newText: "new"});
  assert.equal((multiple.details as Record<string, unknown>).code, "EDIT_OLD_TEXT_NOT_UNIQUE"); assert.equal((multiple.details as Record<string, unknown>).resolved_target, join(scratch, "driver.py"));
  const denied = await call("write", {path: join(root, "link/driver.py"), content: "overwrite"});
  assert.equal((denied.details as Record<string, unknown>).code, "PATH_SCOPE_DENIED"); assert.match(JSON.stringify(denied.content), /escapes workspace/);
  assert.equal(readFileSync(join(scratch, "driver.py"), "utf8"), "duplicate duplicate");
  const good = await call("edit", {path: join(scratch, "driver.py"), oldText: "duplicate duplicate", newText: "new"});
  assert.equal(good.isError, undefined); assert.equal(readFileSync(join(scratch, "driver.py"), "utf8"), "new");
 } finally {rmSync(root, {recursive: true, force: true}); rmSync(scratch, {recursive: true, force: true});}
});

test("empty oldText is rejected before any mutation; file unchanged", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-empty-"));
 try {
  const root = join(base, "root"); mkdirSync(root);
  writeFileSync(join(root, "a.txt"), "ab");
  const tools = buildWorkspacePackTools({root, mode: () => "SIMULATION", bwrapPath: () => null});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  const res = await call("edit", {path: "a.txt", oldText: "", newText: "X"});
  assert.equal(res.isError, true);
  assert.equal((res.details as Record<string, unknown>).code, "EDIT_OLD_TEXT_EMPTY");
  assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "ab");
 } finally {rmSync(base, {recursive: true, force: true});}
});

test("write reports UTF-8 byte length, not UTF-16 code units", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-utf8-"));
 try {
  const root = join(base, "root"); mkdirSync(root);
  const tools = buildWorkspacePackTools({root, mode: () => "SIMULATION", bwrapPath: () => null});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  const res = await call("write", {path: "u.txt", content: "汉字"});
  assert.equal(res.isError, undefined);
  assert.match(JSON.stringify(res.content), /\(6 bytes\)/);
  assert.equal(readFileSync(join(root, "u.txt"), "utf8"), "汉字");
 } finally {rmSync(base, {recursive: true, force: true});}
});
