import './fixtures/pi-lifecycle-owned-bootstrap.js';
import assert from "node:assert/strict";
import { before, after, test } from "node:test";
import { createServer } from "node:net";
import { join } from "node:path";
import { createAssistantMessageEventStream, InMemoryCredentialStore, type Model, type AssistantMessage } from "@earendil-works/pi-ai";
import { createAgentSession, createAgentSessionRuntime, createAgentSessionServices, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager, InteractiveMode } from "@earendil-works/pi-coding-agent";
import type { Terminal } from "@earendil-works/pi-tui";
import { createRosclawExtension, ownedUIState } from "../src/extension/index.js";
import { ActiveSessionContext } from "../src/session/active-context.js";
import { buildProcessTools } from "../src/tools/process-tools.js";
import { attachOwnedUIAbort } from "../src/ui-owned-cancellation.js";
import { contentHash, type MirrorEvent } from "../src/extension/event-mirror.js";

type BatchEvent = MirrorEvent & { mirror_id: string; identity_protocol: string };
type BatchAck = { ok: true; stored: number };
const mirrorBatches: BatchEvent[][] = [];

// This fixture supplies facts, never handlers, gates, watcher internals, or SDK methods.
const home = process.env.PI_LIFECYCLE_OWNED_HOME!;
const agentDir = process.env.PI_CODING_AGENT_DIR!;
const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));
async function until(predicate: () => boolean, label: string, ms = 4500) {
 const end = performance.now() + ms;
 while (!predicate() && performance.now() < end) await delay(10);
 assert.ok(predicate(), label);
}
class InjectedTerminal implements Terminal {
 input?: (data: string) => void;
 columns = 100; rows = 30; kittyProtocolActive = false;
 start(input: (data: string) => void) { this.input = input; }
 stop() { this.input = undefined; }
 async drainInput() {}
 write(_data: string) {} moveBy(_n: number) {} hideCursor() {} showCursor() {}
 clearLine() {} clearFromCursor() {} clearScreen() {} setTitle(_s: string) {}
 setProgress(_v: boolean) {} setProgramStatus(_s: Parameters<Terminal["setProgramStatus"]>[0]) {}
 emit(data: string) { assert.ok(this.input); this.input(data); }
}
const terminal = new InjectedTerminal();
const active = new ActiveSessionContext({ sessionId: "", missionId: "fixture-mission", contextRevision: 7,
 mode: "SIMULATION", profile: "developer", contextState: "LOADING", leaseState: "NONE", actionsAllowed: false });
const operations = new Map<string, Record<string, any>>();
const calls: { method: string; params: Record<string, any> }[] = [];
const undeclared: string[] = [];
let currentId = "consumed", turn = "turn-consumed";
let revision = 7;
const center = {
 isSimAutoPolicy: true,
 snapshot: () => ({ model: "Offline fixture", kernel: "OFFLINE", operator: "OFFLINE", readiness: "BLOCKED", mode: "SIMULATION", context: "STALE", lease: "NONE" }),
 noteWorkspace(_s?: string) {}, noteModel(_s: string) {}, noteProviderOk() {}, noteProviderPaused(_s: string) {},
 subscribe(_fn: () => void) { return () => {}; },
 async bootstrap() {}, async probeOperator(_force?: boolean) {}, async refreshCapabilities(_force?: boolean) {}, async refreshRobotInfo(_force?: boolean) {},
 async call(method: string, params: Record<string, any> = {}): Promise<Record<string, any>> {
  calls.push({ method, params: structuredClone(params) });
  switch (method) {
   case "pi.events.batch": {
    assert.deepEqual(Object.keys(params), ["events"]);
    assert.ok(Array.isArray(params.events), "original mirror supplies a batch");
    const events = params.events as BatchEvent[];
    assert.ok(events.length > 0 && events.length <= 256, "bounded original flush");
    for (const event of events) {
     assert.equal(event.pi_session_id, active.current.sessionId);
     assert.equal(event.mission_id, "fixture-mission");
     assert.equal(event.identity_protocol, "stable-entry/v1");
     assert.match(event.mirror_id, /^mir_[0-9a-f-]{12}$/);
     assert.match(event.content_hash ?? "", /^sha256:[0-9a-f]{64}$/);
     assert.ok(Number.isFinite(Date.parse(event.occurred_at)));
     assert.ok(event.event_type === "message_end" || event.event_type === "turn_end");
     assert.ok(!("text" in event) && !("content" in event), "hash-only mirror");
    }
    mirrorBatches.push(structuredClone(events));
    const ack: BatchAck = { ok: true, stored: events.length };
    return ack;
   }
   case "pi.tools.execute": {
    // Consumer contract of the ORIGINAL executeVia; no tool or SDK handler is replaced.
    assert.deepEqual(Object.keys(params), ["request"]);
    const request = params.request;
    assert.ok(request && typeof request === "object");
    assert.deepEqual(Object.keys(request).sort(), ["schema_version", "request_id", "pi_session_id", "mission_id",
     "context_revision", "body_hash", "mode", "tool_name", "arguments", "requested_at", "idempotency_key", "actor"].sort());
    assert.equal(request.schema_version, "rosclaw.pi_tool_request.v1");
    assert.match(request.request_id, /^ptr_\d+_\d+$/);
    assert.equal(request.pi_session_id, active.current.sessionId);
    assert.equal(request.mission_id, active.current.missionId);
    assert.equal(request.context_revision, active.current.contextRevision);
    assert.equal(request.body_hash, active.current.bodyHash ?? "");
    assert.equal(request.mode, "SIMULATION");
    assert.ok(Number.isFinite(Date.parse(request.requested_at)));
    assert.ok(request.idempotency_key.startsWith(`idem_${active.current.sessionId}_`));
    assert.deepEqual(request.actor, { engine: "pi", process_id: process.pid, uid: process.getuid?.() ?? 0 });
    assert.ok(request.arguments && typeof request.arguments === "object");
    switch (request.tool_name) {
     case "rosclaw_process_start": {
      assert.deepEqual(request.arguments, { command: "fixture-only", cwd: home });
      assert.ok(!operations.has(currentId), "unique fixture operation start");
      const operation = { mission_id: active.current.missionId, session_ref: active.current.sessionId,
       task_id: `task-${currentId}`, operation_id: currentId, turn_id: turn, revision: 7, state: "RUNNING" };
      operations.set(currentId, operation);
      return { ok: true, result: { ok: true, summary: "offline operation started", operation: structuredClone(operation) } };
     }
     case "rosclaw_process_status": {
      assert.deepEqual(Object.keys(request.arguments), ["operation_id"]);
      const op = operations.get(request.arguments.operation_id); assert.ok(op, "exact known tool operation");
      return { ok: true, result: { ok: true, summary: "offline operation status", status: op.state, operation: structuredClone(op) } };
     }
     case "rosclaw_process_output": {
      assert.equal(typeof request.arguments.operation_id, "string");
      const op = operations.get(request.arguments.operation_id); assert.ok(op, "exact known output operation");
      return { ok: true, result: { ok: true, summary: "FINAL fixture output", status: op.state,
       operation: structuredClone(op), stdout: "FINAL fixture output", stderr: "", exit_code: 0 } };
     }
     default: undeclared.push(`pi.tools.execute:${String(request.tool_name)}`);
      throw new Error(`UNDECLARED_FIXTURE_TOOL:${String(request.tool_name)}`);
    }
   }
   case "pi.input.dispatched": {
    // InputController.persist calls this after the exact successful persist.
    assert.deepEqual(Object.keys(params).sort(), ["message_id", "mission_id"]);
    assert.equal(params.mission_id, active.current.missionId);
    assert.equal(typeof params.message_id, "string");
    assert.match(params.message_id, /^msg_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    const persisted = calls.find(c => c.method === "pi.input.persist" && c.params.message_id === params.message_id);
    assert.ok(persisted, "dispatch acknowledges a recorded fixture input only");
    assert.equal(persisted.params.mission_id, params.mission_id);
    assert.equal(persisted.params.session_ref, active.current.sessionId);
    assert.equal(persisted.params.backend_native_id, active.current.sessionId);
    assert.equal(typeof persisted.params.text, "string");
    assert.equal(persisted.params.force_new, false);
    return { ok: true, message_id: params.message_id, fixture_provenance: "offline-input-dispatch-ack-only" };
   }
   case "pi.context": {
    // fetchEmbodiedContext consumes ok/error first; no fabricated envelope or lease.
    assert.deepEqual(Object.keys(params).sort(), ["mission_id", "pi_session_id"]);
    assert.equal(params.mission_id, active.current.missionId);
    assert.equal(params.pi_session_id, active.current.sessionId);
    assert.equal(typeof params.pi_session_id, "string");
    assert.ok(params.pi_session_id.length > 0);
    return { ok: false, error: "FIXTURE_CONTEXT_UNAVAILABLE", fixture_provenance: "offline-context-rejection-no-lease" };
   }
   case "pi.kernel.active": {
    // InputController.activeTaskId reads task?.task_id for the provider-error card.
    assert.deepEqual(Object.keys(params).sort(), ["mission_id", "session_ref"]);
    assert.equal(params.mission_id, active.current.missionId);
    assert.equal(params.session_ref, active.current.sessionId);
    assert.equal(typeof params.session_ref, "string");
    assert.ok(params.session_ref.length > 0);
    const operation = operations.get(currentId);
    if (!operation) return { ok: true, task: null, fixture_provenance: "offline-no-current-operation-task" };
    assert.equal(operation.mission_id, params.mission_id);
    assert.equal(operation.session_ref, params.session_ref);
    assert.equal(operation.operation_id, currentId);
    assert.equal(operation.task_id, `task-${currentId}`);
    assert.equal(operation.turn_id, `turn-${currentId}`);
    assert.equal(operation.revision, 7);
    assert.ok(Number.isInteger(revision) && revision >= operation.revision);
    assert.ok(["RUNNING", "SUCCEEDED"].includes(operation.state));
    return { ok: true, task: operation.state === "RUNNING" ? {
     task_id: operation.task_id, mission_id: operation.mission_id, session_ref: operation.session_ref,
     state: "RUNNING", active_revision: revision,
    } : null, fixture_provenance: "offline-current-operation-task-projection-not-kernel-evidence" };
   }
   case "pi.capability.snapshot": return { ok: false };
   case "pi.context.get": return { ok: false, code: "FIXTURE_CONTEXT_UNAVAILABLE" };
   case "pi.turn.record": return { ok: true };
   case "pi.turn.latest": return { ok: true, turn: { turn_id: turn } };
   case "pi.input.persist": return { ok: true };
   case "pi.task.ensure_effect": return { ok: true, task_id: `task-${currentId}`, revision: 7 };
   case "pi.op.start": {
    const operation = { mission_id: active.current.missionId, session_ref: active.current.sessionId,
     task_id: `task-${currentId}`, operation_id: currentId, turn_id: turn, revision: 7, state: "RUNNING" };
    operations.set(currentId, operation);
    return { ok: true, operation: structuredClone(operation) };
   }
   case "pi.op.get": {
    const op = operations.get(params.operation_id); assert.ok(op, "exact known operation");
    return { ok: true, status: op.state, operation: structuredClone(op) };
   }
   case "pi.op.output": {
    const op = operations.get(params.operation_id); assert.ok(op);
    return { ok: true, status: op.state, operation: structuredClone(op), stdout: "FINAL fixture output", stderr: "", exit_code: 0 };
   }
   case "pi.kernel.events": return { ok: true, events: [] };
   case "pi.kernel.get": return { ok: true, task: { task_id: params.task_id, state: "RUNNING", active_revision: revision } };
   case "pi.task.active": return { ok: true, task: null };
   default: undeclared.push(method); throw new Error(`UNDECLARED_CENTER_METHOD:${method}`);
  }
 },
};
const model: Model<"openai-completions"> = { id: "lifecycle", name: "Offline lifecycle", api: "openai-completions", provider: "lifecycle-fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 100 };
type ToolArguments = Extract<AssistantMessage["content"][number], { type: "toolCall" }>["arguments"];
type Step = { tool: string; args: ToolArguments; id: string } | "done" | "hold" | "provider-error";
let steps: Step[] = [];
let release: (() => void) | undefined;
let streams = 0;
const ended: any[] = [], toolEnds: any[] = [], errors: unknown[] = [];
let settledExtensions = 0;
async function extensionSettled(previous: number, label: string) {
 // Agent idle alone does not certify completion of the queued extension callbacks.
 await until(() => settledExtensions > previous && host.session.isIdle, label);
}
let host: Awaited<ReturnType<typeof createAgentSessionRuntime>>;
let mode: InteractiveMode;
let binding: ReturnType<typeof attachOwnedUIAbort>;
const requests: any[] = [];
const replies = [ { ok: true, code: "CANCELLED" }, { ok: true, code: "ALREADY_SUCCEEDED", operations_cancelled: 0 }, { ok: false, code: "OWNERSHIP_STALE" }, { ok: false, code: "CALLER_MISMATCH" } ];
const peers = new Set<import("node:net").Socket>();
const server = createServer(socket => {
 peers.add(socket); socket.once("close", () => peers.delete(socket));
 let text = "";
 socket.on("data", chunk => {
  text += chunk.toString(); if (!text.endsWith("\n")) return;
  const request = JSON.parse(text); requests.push(request);
  assert.equal(request.method, "pi.op.cancel_owned");
  const expected = operations.get(request.params.operation_id)!; assert.ok(expected);
  for (const key of ["mission_id", "session_ref", "task_id", "operation_id", "turn_id"]) assert.equal(request.params[key], expected[key]);
  assert.equal(request.params.token, process.env.ROSCLAW_CONTROL_TOKEN);
  assert.ok(requests.length <= 4);
  socket.end(JSON.stringify(replies[requests.length - 1]) + "\n");
 });
});
before(async () => {
 assert.ok(home && agentDir, "operator-owned import environment required");
 // The cancellation server is installed only in the third group: other bridge paths must stay offline.
 const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false }, quietStartup: true });
 const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
  modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
 const streamSimple = (_model: unknown, _context: unknown, options: any) => {
  assert.ok(++streams <= 40, "bounded in-memory model streams");
  const step = steps.shift() ?? "done";
  const stream = createAssistantMessageEventStream();
  const message = (stopReason: AssistantMessage["stopReason"]): AssistantMessage => ({ role: "assistant", api: model.api,
   provider: model.provider, model: model.id, timestamp: Date.now(), stopReason,
   content: typeof step === "object" ? [{ type: "toolCall", id: step.id, name: step.tool, arguments: step.args }] : [{ type: "text", text: "offline fixture reply" }],
   usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  const finish = () => stream.push({ type: "done", reason: typeof step === "object" ? "toolUse" : "stop", message: message(typeof step === "object" ? "toolUse" : "stop") });
  if (step === "hold") {
   release = finish;
   options.signal?.addEventListener("abort", () => { release = undefined; stream.push({ type: "error", reason: "aborted", error: message("aborted") }); }, { once: true });
  } else if (step === "provider-error") stream.push({ type: "error", reason: "error", error: { ...message("error"), errorMessage: "model request cancelled" } });
  else finish();
  return stream;
 };
 runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "SYNTHETIC_OFFLINE", models: [model], streamSimple });
 const manager = SessionManager.inMemory(home);
 active.patch({ sessionId: manager.getSessionId() });
 const lateSession: { session?: { setActiveToolsByName(names: string[]): void } } = {};
 const original = createRosclawExtension({ profile: "developer", version: "source-fixture", systemPrompt: "offline", rosclawHome: home,
  active, center: center as never, locale: { effective: "en", subscribe: () => () => {} } as never,
  coordinator: { setNotify() {}, async bindSession() {} } as never,
  taskContext: { workspaceRoot: home, workspaceSource: "default" } as never,
  osIsolationProbe: () => ({ isolationReady: false }), sessionManager: manager, lateSession,
  allowedModelToolNames: ["process_start", "process_status", "process_output"] });
 const loader = new DefaultResourceLoader({ cwd: home, agentDir, settingsManager: settings, noExtensions: true, noSkills: true,
  noContextFiles: true, noThemes: true, noPromptTemplates: true,
  extensionFactories: [{ name: "rosclaw", factory: original }, { name: "observe-only", factory: pi => {
   pi.on("message_end", event => { ended.push(structuredClone(event.message)); });
   pi.on("agent_end", () => { settledExtensions++; });
   pi.on("tool_execution_end", event => {
    toolEnds.push(structuredClone(event));
    if (event.toolCallId === "call-consumed") {
     // This factory runs AFTER the original extension's ownership/track handler,
     // not in the earlier public subscription. Only advance a fixture fact.
     assert.equal(event.isError, false);
     assert.equal((event.result.details as any).operation.operation_id, "consumed");
     assert.deepEqual(ownedUIState(home).owners.get(active.current.sessionId), operations.get("consumed"));
     operations.get("consumed")!.state = "SUCCEEDED";
    }
   });
  } }] });
 await loader.reload(); assert.deepEqual(loader.getExtensions().errors, []);
 host = await createAgentSessionRuntime(async target => {
  const services = await createAgentSessionServices({ cwd: home, agentDir, modelRuntime: runtime, settingsManager: settings });
  const { session, extensionsResult } = await createAgentSession({ ...target, cwd: home, agentDir, modelRuntime: runtime,
   model: runtime.getModel(model.provider, model.id)!, settingsManager: settings, resourceLoader: loader, thinkingLevel: "off",
   tools: ["process_start", "process_status", "process_output"], customTools: buildProcessTools({ rosclawHome: home, active, center: center as never, sessionCwd: home }) });
  lateSession.session = session;
  session.subscribe(event => { if (event.type === "message_end" && event.message.role === "toolResult") {
   const message = event.message;
   const saved = session.messages.find(m => m.role === "toolResult" && m.toolCallId === message.toolCallId);
   try { assert.deepEqual(saved, message); } catch (e) { errors.push(e); }
  } });
  return { session, services, extensionsResult, diagnostics: [] };
 }, { cwd: home, agentDir, sessionManager: manager });
 mode = new InteractiveMode(host, { terminal, verbose: false });
 binding = attachOwnedUIAbort(mode, { rosclawHome: home,
  current: () => ownedUIState(home).owners.get(active.current.sessionId),
  turn: () => ownedUIState(home).turns.get(active.current.sessionId) });
 await mode.init();
});
after(async () => {
 release?.();
 if (host) await host.dispose();
 // Stop only the injected TUI; this is not the main.ts shutdown bridge.
 (mode as unknown as { tui: { stop(): void } })?.tui?.stop(); terminal.stop();
 if (server.listening) await new Promise<void>((resolve, reject) => server.close(e => e ? reject(e) : resolve()));
 await until(() => peers.size === 0, "all owned IPC peers closed", 500);
});
function start(id: string, final: Step = "done") {
 currentId = id; turn = `turn-${id}`;
 steps = [{ tool: "process_start", args: { command: "fixture-only", cwd: home }, id: `call-${id}` }, final];
 return host.session.prompt(`fixture ${id}`);
}
function customResults() { return host.session.messages.filter(m => m.role === "custom" && m.customType === "rosclaw.operation.result") as any[]; }
function assertOriginalMirrorFlush() {
 const events = mirrorBatches.flat();
 const messages = ended.filter((m): m is AssistantMessage => m.role === "assistant");
 const mirrored = events.filter(e => e.event_type === "message_end");
 assert.ok(messages.length > 0, "genuine SDK assistant events observed");
 assert.equal(mirrored.length, messages.length, "every original message_end flushed exactly once");
 assert.equal(new Set(events.map(e => e.mirror_id)).size, events.length, "no ACKed batch replay");
 assert.equal(calls.filter(c => c.method === "pi.events.batch").length, mirrorBatches.length);
 for (const [index, message] of messages.entries()) {
  const event = mirrored[index]!;
  assert.equal(event.content_hash, contentHash(JSON.stringify(message.content ?? "")));
  assert.equal(event.model, message.model);
  const responseId = (message as AssistantMessage & { responseId?: string }).responseId;
  assert.equal(event.pi_entry_id, responseId || "");
  assert.deepEqual(event.usage, { ...message.usage,
   _rosclaw_terminal: { stopReason: message.stopReason, responseId: responseId || null } });
 }
 const turns = events.filter(e => e.event_type === "turn_end");
 assert.ok(turns.length > 0, "original turn_end callback flushed");
 for (const event of turns) {
  assert.ok(messages.some(m => event.content_hash === contentHash(JSON.stringify(m.content ?? ""))),
   "turn_end hash matches a genuine SDK assistant message");
 }
}

test("FULL_EXTENSION_SDK_FINALIZED_RESULT", async () => {
 currentId = "consumed"; turn = "turn-consumed";
 steps = [{ tool: "process_start", args: { command: "fixture-only", cwd: home }, id: "call-consumed" },
  { tool: "process_status", args: { operation_id: "consumed" }, id: "status-consumed" }, "done"];
 // The later inline observation factory advances the fact after original ownership tracking.
 await host.session.prompt("fixture consumed");
 const saved = host.session.messages.filter(m => m.role === "toolResult") as any[];
 const result = saved.find(m => m.toolCallId === "status-consumed"); assert.ok(result); assert.equal(result.isError, false);
 assert.equal(result.toolName, "process_status"); assert.equal(result.details.status, "SUCCEEDED");
 assert.equal(result.details.operation.operation_id, "consumed"); assert.equal(result.details.operation.task_id, "task-consumed");
 assert.equal(result.details.operation.revision, 7); assert.equal(result.details.operation.session_ref, active.current.sessionId);
 assert.equal(result.details.operation.mission_id, "fixture-mission"); assert.equal(result.details.operation.turn_id, "turn-consumed");
 assert.deepEqual(result, ended.find(m => m.role === "toolResult" && m.toolCallId === "status-consumed"));
 assert.deepEqual(result.details, toolEnds.find(e => e.toolCallId === "status-consumed").result.details);
 assertOriginalMirrorFlush();
 assert.deepEqual(ownedUIState(home).owners.get(active.current.sessionId), operations.get("consumed") && { ...operations.get("consumed"), state: "RUNNING" });
 assert.deepEqual(errors, []);
});

test("BUSY_TO_IDLE_EXTENSION_WAKEUP", async () => {
 const running = start("wake", "hold");
 await until(() => !!release, "real second stream is held");
 operations.get("wake")!.state = "SUCCEEDED";
 const baseline = customResults().length;
 await until(() => calls.some(c => c.method === "pi.op.get" && c.params.operation_id === "wake"), "original watcher observed busy terminal");
 assert.equal(host.session.isStreaming, true); assert.equal(customResults().length, baseline);
 release!(); release = undefined; await running;
 await until(() => customResults().some(m => m.details.operation_id === "wake"), "idle original extension starts managed turn");
 await until(() => host.session.isIdle, "managed result turn settles");
 const wake = customResults().filter(m => m.details.operation_id === "wake"); assert.equal(wake.length, 1);
 assert.deepEqual(wake[0].details, { operation_id: "wake", state: "SUCCEEDED" });
 assert.match(wake[0].content, /process_output/);
 assert.ok(calls.some(c => c.method === "pi.kernel.get" && c.params.task_id === "task-wake"));
 // A different operation with the same old revision is archived, not new work.
 revision = 8;
 await start("stale"); operations.get("stale")!.state = "SUCCEEDED";
 await until(() => calls.some(c => c.method === "pi.kernel.get" && c.params.task_id === "task-stale"), "stale revision ownership rechecked");
 await delay(100);
 assert.equal(customResults().filter(m => m.details.operation_id === "stale").length, 0);
 assert.equal(customResults().filter(m => m.details.operation_id === "consumed").length, 0);
 assert.equal(customResults().filter(m => m.details.operation_id === "wake").length, 1);
 revision = 7;
});

test("OWNER_CANCEL_CAUSE_BOUNDARY", async () => {
 await new Promise<void>(resolve => server.listen(join(home, "run", "pi-bridge.sock"), resolve));
 terminal.emit("nonempty editor");
 const editor = (mode as unknown as { defaultEditor: { getText(): string } }).defaultEditor;
 assert.ok(editor.getText(), "PI_LIFECYCLE_ASSERT_001"); terminal.emit("\u0003"); assert.equal(editor.getText(), "", "PI_LIFECYCLE_ASSERT_002");
 assert.deepEqual(await binding.drain(), [], "PI_LIFECYCLE_ASSERT_003"); assert.equal(requests.length, 0, "PI_LIFECYCLE_ASSERT_004");
 for (const [index, id] of ["cancel", "success", "stale-cancel", "foreign-cancel"].entries()) {
  const previousSettlement = settledExtensions;
  const running = start(id, "hold"); await until(() => !!release, "real stream ready for ESC");
  assert.equal(host.session.isStreaming, true, "PI_LIFECYCLE_ASSERT_005");
  const owner = ownedUIState(home).owners.get(active.current.sessionId)!;
  assert.equal(owner.operation_id, id, "PI_LIFECYCLE_ASSERT_006"); assert.equal(owner.turn_id, `turn-${id}`, "PI_LIFECYCLE_ASSERT_007");
  terminal.emit("\u001b"); await running;
  // Abort/prompt completion is not extension lifecycle settlement. Let the
  // original SDK finish message/turn callbacks before checking or advancing input.
  await extensionSettled(previousSettlement, "original editor-aborted lifecycle settles");
  const outcome = await binding.drain();
  assert.deepEqual(outcome, [{ ok: replies[index].ok, code: replies[index].code }], "PI_LIFECYCLE_ASSERT_008");
  await until(() => peers.size === 0, "previous cancellation connection closed", 500);
  assert.equal(requests.length, index + 1, "PI_LIFECYCLE_ASSERT_009");
  if (index === 1) {
   // Exact ALREADY_SUCCEEDED identity is quiescent even on another real streaming ESC.
   const previousDuplicate = settledExtensions;
   steps = ["hold"]; const duplicate = host.session.prompt("fixture duplicate input", { source: "extension" });
   await until(() => !!release, "duplicate stream"); terminal.emit("\u001b"); await duplicate;
   await extensionSettled(previousDuplicate, "original duplicate-aborted lifecycle settles");
   assert.deepEqual(await binding.drain(), [], "PI_LIFECYCLE_ASSERT_010"); assert.equal(requests.length, 2, "PI_LIFECYCLE_ASSERT_011");
  }
 }
 // A programmatic/provider cancellation is not editor input and cannot acquire owned cancel authority.
 const previousProgrammatic = settledExtensions;
 steps = ["hold"]; const programmatic = host.session.prompt("programmatic fixture", { source: "extension" });
 await until(() => !!release, "programmatic stream"); await host.session.abort(); await programmatic;
 await extensionSettled(previousProgrammatic, "original programmatic-aborted lifecycle settles");
 assert.deepEqual(await binding.drain(), [], "PI_LIFECYCLE_ASSERT_012"); assert.equal(requests.length, 4, "PI_LIFECYCLE_ASSERT_013");
 const previousProvider = settledExtensions;
 steps = ["provider-error"]; await host.session.prompt("provider failure fixture", { source: "extension" });
 await extensionSettled(previousProvider, "original provider-error lifecycle settles");
 assert.deepEqual(await binding.drain(), [], "PI_LIFECYCLE_ASSERT_014"); assert.equal(requests.length, 4, "PI_LIFECYCLE_ASSERT_015");
 assert.ok(!calls.some(c => c.method === "pi.session.interrupt" || c.method === "pi.op.cancel_owned"), "PI_LIFECYCLE_ASSERT_016");
 assertOriginalMirrorFlush();
 assert.deepEqual(undeclared, [], "unknown fixture method must fail, never generic ok");
 assert.deepEqual(errors, [], "PI_LIFECYCLE_ASSERT_017");
});
