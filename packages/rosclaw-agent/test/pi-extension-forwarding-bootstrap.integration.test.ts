import './fixtures/pi-lifecycle-owned-bootstrap.js';
import { establishOwnedFixture, lifecycleFixture, retainLifecycleFixture, releaseLifecycleFixture } from "./fixtures/pi-lifecycle-owned-bootstrap.js";
import assert from "node:assert/strict";
import { before, after, test } from "node:test";
import { existsSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { createRosclawExtension } from "../src/extension/index.js";
import { EventMirror, contentHash, type MirrorEvent } from "../src/extension/event-mirror.js";
import { ActiveSessionContext } from "../src/session/active-context.js";
import { ProductStateCenter } from "../src/session/state-center.js";

before(() => retainLifecycleFixture());
after(() => releaseLifecycleFixture());
type Batch = MirrorEvent & { mirror_id: string; identity_protocol: string };
const digest = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
function activeContext(missionId?: string) {
 return new ActiveSessionContext({ sessionId: "forward-session", missionId, contextRevision: 7,
  mode: "SIMULATION", profile: "developer", contextState: "FRESH", leaseState: "NONE", actionsAllowed: false });
}
type Callback = (event: any, ctx: any) => unknown;
// Registration-only fixture, NOT a substitute SDK. The original three groups run the real SDK.
// No default/Proxy method: an undeclared factory API access must fail.
function forwarding(center: { call(method: string, params?: Record<string, unknown>): Promise<Record<string, unknown>> }, active = activeContext("forward-mission")) {
 const handlers = new Map<string, Callback[]>();
 const api = {
  on(name: string, callback: Callback) { const list = handlers.get(name) ?? []; list.push(callback); handlers.set(name, list); },
  registerCommand(_name: string, _spec: unknown) {},
  registerShortcut(_key: string, _spec: unknown) {},
  registerMessageRenderer(_name: string, _renderer: unknown) {},
  registerEntryRenderer(_name: string, _renderer: unknown) {},
 };
 createRosclawExtension({ profile: "developer", version: "source-forwarding", systemPrompt: "offline",
  rosclawHome: lifecycleFixture.home, active, center: center as never,
  locale: { effective: "en" } as never, coordinator: { setNotify() {} } as never,
  taskContext: { workspaceRoot: lifecycleFixture.home, workspaceSource: "default" } as never,
  osIsolationProbe: () => ({ isolationReady: false }) })(api as never);
 // Last registered message_end is the original mirror callback; no handler is replaced.
 const message = handlers.get("message_end")!;
 assert.ok(message.length > 1);
 const shutdown = handlers.get("session_shutdown")!;
 assert.ok(shutdown.length > 1);
 return {
  active,
  emit(id: string, text = id) { return message.at(-1)!({ message: { role: "assistant", responseId: id,
   content: [{ type: "text", text }], model: "offline-forwarding", stopReason: "stop", usage: { input: 2 } } }, { hasUI: false }); },
  flush() { return shutdown.at(-1)!({}, { hasUI: false }); },
 };
}
function events(params: Record<string, unknown> | undefined): Batch[] {
 assert.ok(params); assert.deepEqual(Object.keys(params), ["events"]);
 assert.ok(Array.isArray(params.events));
 return params.events as Batch[];
}

test("PI_PORTABLE_BOOTSTRAP_BEFORE_SDK", async () => {
 assert.equal(process.env.PI_CODING_AGENT_DIR, lifecycleFixture.agentDir);
 assert.equal(process.env.PI_OFFLINE, "1"); assert.equal(process.env.PI_TELEMETRY, "0");
 // Dynamic import is after the first side-effect bootstrap. More importantly, the SDK
 // already evaluated through index.ts: its import-time TOOLS_DIR must equal this bin.
 const sdkEntry = import.meta.resolve("@earendil-works/pi-coding-agent");
 const tools = await import(new URL("./utils/tools-manager.js", sdkEntry).href) as { getToolPath(name: string): string | null };
 for (const name of ["rg", "fd"]) assert.equal(tools.getToolPath(name), join(lifecycleFixture.agentDir, "bin", name));
 const saved = process.env.PI_CODING_AGENT_DIR;
 try {
  process.env.PI_CODING_AGENT_DIR = join(lifecycleFixture.home, "must-not-be-cached");
  assert.equal(tools.getToolPath("rg"), join(lifecycleFixture.agentDir, "bin", "rg"), "SDK tools path was cached only after bootstrap");
 } finally { process.env.PI_CODING_AGENT_DIR = saved; }
 const rg = join(lifecycleFixture.agentDir, "bin", "rg"), fd = join(lifecycleFixture.agentDir, "bin", "fd");
 const beforeHashes = [digest(rg), digest(fd)];
 const env = { PI_LIFECYCLE_OWNED_HOME: lifecycleFixture.home, PI_CODING_AGENT_DIR: lifecycleFixture.agentDir,
  PI_OFFLINE: "old", PI_TELEMETRY: "old" };
 const originalEnv = { ...env };
 // A local copy is validated against its own provenance, not the worker's pins.
 // Supplying its directories as a worker must still enforce the original pins.
 const workerHashes = ["e36d0eb52e70696bdf1781392722e05a21bb91d3b7b762ef5ec20e5df2ec687b",
  "b96a9c5eb1f619efe836e7d5176a9aa15d6643f11c46cda6709ff8abb43eeed4"];
 const matchesWorkerPins = beforeHashes.every((hash, index) => hash === workerHashes[index]);
 if (lifecycleFixture.workerOwned) assert.equal(matchesWorkerPins, true);
 if (!lifecycleFixture.workerOwned) {
  const localProvenance = JSON.parse(readFileSync(join(lifecycleFixture.agentDir, "tool-provenance.json"), "utf8"));
  for (const [index, name] of (["rg", "fd"] as const).entries()) {
   assert.equal(localProvenance[name].sha256, beforeHashes[index]);
   assert.equal(localProvenance[name].size, readFileSync(join(lifecycleFixture.agentDir, "bin", name)).length);
  }
 }
 if (matchesWorkerPins) {
  const adopted = establishOwnedFixture(env);
  try {
   assert.equal(adopted.workerOwned, true); assert.equal(adopted.home, lifecycleFixture.home);
   assert.equal(env.PI_OFFLINE, "1"); assert.equal(env.PI_TELEMETRY, "0");
  } finally { adopted.dispose(); adopted.dispose(); }
 } else {
  assert.throws(() => establishOwnedFixture(env), /PI_BOOTSTRAP_REAL_TOOL_PIN/);
 }
 assert.deepEqual(env, originalEnv); assert.deepEqual([digest(rg), digest(fd)], beforeHashes);
 assert.ok(existsSync(lifecycleFixture.home), "adoption or rejected adoption must preserve source files");
 const localEnv: Record<string, string | undefined> = { PI_OFFLINE: "prior" };
 const local = establishOwnedFixture(localEnv, { rg, fd });
 const localHome = local.home;
 assert.equal(local.workerOwned, false); assert.notEqual(localHome, lifecycleFixture.home);
 assert.equal(localEnv.PI_CODING_AGENT_DIR, local.agentDir);
 assert.deepEqual([digest(join(local.agentDir, "bin", "rg")), digest(join(local.agentDir, "bin", "fd"))], beforeHashes);
 local.dispose(); local.dispose();
 assert.equal(existsSync(localHome), false); assert.deepEqual(localEnv, { PI_OFFLINE: "prior" });
 assert.throws(() => establishOwnedFixture({ PI_LIFECYCLE_OWNED_HOME: lifecycleFixture.home }), /PI_BOOTSTRAP_PARTIAL_WORKER_ENV/);
 assert.throws(() => establishOwnedFixture({}, { rg: join(lifecycleFixture.home, "missing-rg"), fd }), /ENOENT/);
 const fs = await import("node:fs");
 const savedPath = process.env.PATH;
 const discoveredEnv: Record<string, string | undefined> = { PI_TELEMETRY: "prior", ROSCLAW_HOME: "prior-home" };
 const discoveredBefore = { ...discoveredEnv };
 // SOURCE has genuine staged bytes but no distro tools. It exercises explicit
 // source copying only; unpreloaded distro npm must exercise default discovery.
 const discovered = establishOwnedFixture(discoveredEnv, lifecycleFixture.workerOwned ? { rg, fd } : undefined);
 const discoveredHome = discovered.home;
 try {
  assert.equal(discovered.workerOwned, false);
  assert.equal(discoveredEnv.PI_OFFLINE, "1"); assert.equal(discoveredEnv.PI_TELEMETRY, "0");
  assert.equal(discoveredEnv.PI_CODING_AGENT_DIR, discovered.agentDir);
  const provenance = JSON.parse(readFileSync(join(discovered.agentDir, "tool-provenance.json"), "utf8")) as
   Record<"rg" | "fd", { source: string; canonical: string; size: number; sha256: string }>;
  for (const name of ["rg", "fd"] as const) {
   const record = provenance[name], target = join(discovered.agentDir, "bin", name);
   if (lifecycleFixture.workerOwned) assert.equal(record.source, name === "rg" ? rg : fd);
   else assert.match(record.source, name === "fd" ? /^\/(usr\/bin|usr\/local\/bin|bin)\/(fd|fdfind)$/ : /^\/(usr\/bin|usr\/local\/bin|bin)\/rg$/);
   assert.equal(fs.realpathSync(record.source), record.canonical);
   assert.equal(fs.realpathSync(record.canonical), record.canonical);
   assert.equal(fs.lstatSync(record.canonical).isFile(), true);
   assert.equal(fs.lstatSync(target).isFile(), true); assert.equal(fs.lstatSync(target).isSymbolicLink(), false);
   assert.equal(fs.lstatSync(target).mode & 0o777, 0o555);
   assert.equal(fs.lstatSync(target).size, record.size);
   assert.equal(digest(record.canonical), record.sha256); assert.equal(digest(target), record.sha256);
   assert.deepEqual([...readFileSync(target).subarray(0, 4)], [0x7f, 0x45, 0x4c, 0x46]);
  }
  // Exercise the Debian alias with genuine bytes, not a fake/version-only executable.
  const alias = join(discoveredHome, "fdfind");
  fs.copyFileSync(provenance.fd.canonical, alias, fs.constants.COPYFILE_EXCL); fs.chmodSync(alias, 0o555);
  const aliasEnv: Record<string, string | undefined> = {};
  const aliased = establishOwnedFixture(aliasEnv, { rg: provenance.rg.canonical, fd: alias });
  const aliasHome = aliased.home;
  try {
   const aliasRecord = JSON.parse(readFileSync(join(aliased.agentDir, "tool-provenance.json"), "utf8"));
   assert.equal(aliasRecord.fd.source, alias); assert.equal(aliasRecord.fd.canonical, alias);
   assert.equal(aliasRecord.fd.sha256, provenance.fd.sha256);
   assert.equal(digest(join(aliased.agentDir, "bin", "fd")), provenance.fd.sha256);
   assert.equal(existsSync(join(aliased.agentDir, "bin", "fdfind")), false);
  } finally { aliased.dispose(); aliased.dispose(); }
  assert.equal(existsSync(aliasHome), false); assert.deepEqual(aliasEnv, {});
  const link = join(discoveredHome, "linked-rg"); fs.symlinkSync(provenance.rg.canonical, link);
  const invalidEnv = { PI_OFFLINE: "unchanged" };
  assert.throws(() => establishOwnedFixture(invalidEnv, { rg: link, fd: alias }), /PI_BOOTSTRAP_LOCAL_CANONICAL_TOOL/);
  assert.throws(() => establishOwnedFixture(invalidEnv, { rg: discoveredHome, fd: alias }), /PI_BOOTSTRAP_LOCAL_UNSAFE_TOOL/);
  assert.throws(() => establishOwnedFixture(invalidEnv, { rg: provenance.rg.canonical, fd: join(discoveredHome, "missing-fdfind") }), /ENOENT/);
  assert.deepEqual(invalidEnv, { PI_OFFLINE: "unchanged" });
 } finally { discovered.dispose(); discovered.dispose(); }
 assert.equal(existsSync(discoveredHome), false); assert.deepEqual(discoveredEnv, discoveredBefore);
 assert.equal(process.env.PATH, savedPath);
});

test("PI_FORWARDING_ACK_AND_REJECTION", async () => {
 const batches: Batch[][] = [];
 let ack = false;
 const center = { async call(method: string, params?: Record<string, unknown>) {
  assert.equal(this, center, "seam retains center receiver"); assert.equal(method, "pi.events.batch");
  const batch = events(params); batches.push(structuredClone(batch));
  return ack ? { ok: true, stored: batch.length } : { ok: false, code: "FIXTURE_REJECTED" };
 } };
 const original = forwarding(center);
 await original.emit("ack-one");
 assert.equal(batches.length, 1); assert.equal(batches[0]!.length, 1);
 assert.equal(batches[0]![0]!.content_hash, contentHash(JSON.stringify([{ type: "text", text: "ack-one" }])));
 assert.equal(batches[0]![0]!.pi_session_id, "forward-session");
 assert.equal(batches[0]![0]!.mission_id, "forward-mission");
 assert.ok(!("content" in batches[0]![0]!));
 ack = true; await original.flush();
 assert.deepEqual(batches[1], batches[0], "ok:false preserves exact unACKed payload/identity");
 await original.flush(); await original.emit("ack-one");
 assert.equal(batches.length, 2, "ACKed stable replay is not fresh work");
 await original.emit("ack-two"); assert.equal(batches.length, 3);
 assert.equal(batches[2]![0]!.pi_entry_id, "ack-two");
});

test("PI_FORWARDING_THROW_RECOVERY", async () => {
 const active = activeContext(); // no mission => no unrelated capability probe in this focused unit
 const batches: Batch[][] = [], methods: string[] = [], states: string[] = [];
 let rejectTransport = true;
 let recover: ((value: Record<string, unknown>) => void) | undefined;
 const recovery = new Promise<Record<string, unknown>>(resolve => { recover = resolve; });
 const failure = new Error("FIXTURE_TRANSPORT_FAILURE");
 const center = new ProductStateCenter({ rosclawHome: lifecycleFixture.home, active,
  operatorSocket: join(lifecycleFixture.home, "unused-operator.sock"), productVersion: "source",
  call: async (home, method, params) => {
   assert.equal(home, lifecycleFixture.home); methods.push(method);
   if (method === "pi.status") return recovery;
   assert.equal(method, "pi.events.batch");
   const batch = events(params); batches.push(structuredClone(batch));
   if (rejectTransport) throw failure;
   return { ok: true, stored: batch.length };
  } });
 const unsubscribe = center.subscribe(() => states.push(center.snapshot().kernel));
 try {
  const original = forwarding(center, active);
  await original.emit("throw-one");
  assert.equal(center.snapshot().kernel, "UNREACHABLE"); assert.equal(active.current.contextState, "STALE");
  assert.deepEqual(methods, ["pi.events.batch", "pi.status"]);
  assert.ok(states.includes("UNREACHABLE"));
  recover!({ ok: true }); await recovery; await Promise.resolve();
  assert.equal(center.snapshot().kernel, "READY");
  assert.equal(active.current.contextState, "STALE", "transport recovery does not fabricate fresh context");
  rejectTransport = false; await original.flush();
  assert.deepEqual(batches[1], batches[0]);
  await original.flush(); assert.equal(batches.length, 2);
  assert.ok(states.includes("READY"));
 } finally { recover?.({ ok: true }); unsubscribe(); }
});

test("PI_FORWARDING_REQUEUE_ORDER_OWNERSHIP", async () => {
 const attempts: Batch[][] = [];
 let resolveBatch: ((value: Record<string, unknown>) => void) | undefined;
 let hold = true;
 const mirror = new EventMirror(lifecycleFixture.home, "old-session", "old-mission", async (home, method, params) => {
  assert.equal(home, lifecycleFixture.home); assert.equal(method, "pi.events.batch");
  const batch = events(params); assert.ok(batch.length > 0 && batch.length <= 256);
  attempts.push(structuredClone(batch));
  if (hold) return new Promise(resolve => { resolveBatch = resolve; });
  return { ok: true, stored: batch.length };
 });
 for (let i = 0; i < 600; i++) mirror.push("message_end", { entryId: `old-${i}`, text: `old-${i}` });
 const inFlight = mirror.flush();
 assert.equal(mirror.pending, 600); assert.equal(await mirror.flush(), 0, "no overlapping RPC");
 mirror.retarget("new-session", "new-mission");
 for (let i = 0; i < 400; i++) mirror.push("message_end", { entryId: `new-${i}`, text: `new-${i}` });
 assert.equal(mirror.pending, 1000);
 assert.throws(() => mirror.push("message_end", { entryId: "retry-overflow", text: "later" }), { code: "MIRROR_OVERFLOW" });
 assert.equal(mirror.droppedOverflow, 1);
 resolveBatch!({ ok: false }); assert.equal(await inFlight, 0);
 assert.equal(mirror.pending, 1000); hold = false;
 assert.equal(await mirror.flush(), 1000);
 assert.equal(attempts.length, 5, "failed attempt plus four bounded drain RPCs");
 assert.deepEqual(attempts[1], attempts[0], "failed prefix requeued unchanged ahead of suffix");
 const accepted = attempts.slice(1).flat();
 assert.deepEqual(accepted.map(event => event.pi_entry_id), [
  ...Array.from({ length: 600 }, (_, i) => `old-${i}`), ...Array.from({ length: 400 }, (_, i) => `new-${i}`),
 ]);
 for (const [i, event] of accepted.entries()) {
  assert.equal(event.pi_session_id, i < 600 ? "old-session" : "new-session");
  assert.equal(event.mission_id, i < 600 ? "old-mission" : "new-mission");
  assert.equal(event.identity_protocol, "stable-entry/v1");
 }
 assert.equal(new Set(accepted.map(event => event.mirror_id)).size, 1000);
 assert.equal(mirror.pending, 0);
 mirror.push("message_end", { entryId: "retry-overflow", text: "later" });
 assert.equal(await mirror.flush(), 1, "rejected identity was not registered as ACKed");
 assert.equal(mirror.diagnostics().overflow_dropped, 1); assert.equal(mirror.diagnostics().unconfirmed, true);
 // Continuous production during ACKs must still return after four RPCs, preserving suffix.
 let replenishments = 0;
 const bounded = new EventMirror(lifecycleFixture.home, "s", "m", async (_home, method, params) => {
  assert.equal(method, "pi.events.batch"); const batch = events(params);
  if (++replenishments <= 4) for (let i = 0; i < 256; i++) bounded.push("turn_end", { text: `${replenishments}-${i}` });
  return { ok: true, stored: batch.length };
 });
 for (let i = 0; i < 256; i++) bounded.push("turn_end", { text: `seed-${i}` });
 assert.equal(await bounded.flush(), 1024); assert.equal(replenishments, 4); assert.equal(bounded.pending, 256);
 assert.equal(await bounded.flush(), 256); assert.equal(bounded.pending, 0);
});
