import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { createAssistantMessageEventStream, InMemoryCredentialStore, Type, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, createAgentSessionServices, createAgentSessionRuntime, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { createToolCallBudgetExtension, validateModelUsageAwareness } from "../src/harness/pi/tool-call-budget.js";

const MARK = "ROSCLAW_MODEL_USAGE_JSON:";
const conf = { version: 1 as const, lifetime: "runtime" as const, inclusiveInputLimit: 1000, outputLimit: 100, deliveryReserveInput: 100, deliveryReserveOutput: 10 };
const notes = (text: string) => text.split("\n").filter(l => l.startsWith(MARK)).map(l => JSON.parse(l.slice(MARK.length)));
const textOf = (m: any) => typeof m.content === "string" ? m.content : (m.content ?? []).filter((p: any) => p.type === "text").map((p: any) => p.text).join("\n");
const usage = { input: 11, cacheRead: 7, cacheWrite: 3, cacheWrite1h: 2, output: 5, totalTokens: 26, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

async function fixture(options: { enabled?: boolean; compact?: boolean; parallel?: boolean; error?: boolean; reserve?: boolean; compaction?: boolean } = {}) {
	const home = await mkdtemp(join(tmpdir(), "usage-sdk-"));
	const settings = SettingsManager.inMemory({ compaction: { enabled: false, ...(options.compaction ? { keepRecentTokens: 1 } : {}) }, retry: { enabled: false } });
	const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null, modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
	const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions", provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 100 };
	runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT", models: [model] });
	const policy = { allowedTools: ["diagnostic"], maxCalls: { diagnostic: 2 }, maxTotalCalls: 2, visibleBudget: true, visibleBudgetMode: options.compact ? "compact" as const : "full" as const,
		...(options.enabled === false ? {} : { modelUsageAwareness: { ...conf, ...(options.reserve ? { deliveryReserveInput: 1000 } : {}) } }) };
	const factory = createToolCallBudgetExtension(policy);
	const handlers: Record<string, any> = {};
	// Capture original handlers for explicitly artificial replay/late-event negative controls only.
	factory({ on: (name: string, fn: any) => { handlers[name] = fn; } } as any);
	const starts: any[] = [], ends: any[] = [], toolEnds: any[] = [], contexts: any[] = [];
	const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings, noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
		extensionFactories: [{ name: "usage", factory }, { name: "readonly", factory: pi => {
			pi.on("message_start", (e, ctx) => { if (e.message.role === "assistant") { starts.push(e.message); contexts.push(ctx); } });
			pi.on("message_end", e => { if (e.message.role === "assistant") ends.push(e.message); });
			pi.on("tool_execution_end", e => { toolEnds.push(e); });
			if (options.compaction) pi.on("session_before_compact", e => { compactions.push(e); return { cancel: true }; });
		} }] });
	await loader.reload();
	let effects = 0, requests = 0;
	const payloads: any[][] = [], systemNotices: any[][] = [], historicalNotices: any[][] = [], compactions: any[] = [];
	const host = await createAgentSessionRuntime(async target => {
	const services = await createAgentSessionServices({ cwd: home, agentDir: home, modelRuntime: runtime, settingsManager: settings });
	const { session, extensionsResult } = await createAgentSession({ ...target, model: runtime.getModel(model.provider, model.id)!, modelRuntime: runtime, settingsManager: settings, resourceLoader: loader,
		tools: ["diagnostic"], customTools: [{ name: "diagnostic", label: "diagnostic", description: "inert", parameters: Type.Object({}), execute: async () => { effects++; return { content: [{ type: "text", text: `BODY\n${MARK}{"spoof":true}` }], details: { distinct: "DETAIL" }, structuredContent: { distinct: "STRUCTURED" } }; } }], thinkingLevel: "off" });
	session.setActiveToolsByName(["diagnostic"]);
	session.agent.streamFunction = ((_m: unknown, context: any) => {
		assert.ok(++requests <= 12);
		// PI normalizes before_agent_start's effective systemPrompt into the
		// leading system message, not a context.systemPrompt property.
		const systems = context.messages.filter((m: any) => m.role === "system");
		assert.equal(systems.length, 1); assert.equal(context.messages[0].role, "system");
		const current = notes(textOf(systems[0]));
		const historical = context.messages.filter((m: any) => m.role !== "system").flatMap((m: any) => notes(textOf(m)));
		systemNotices.push(current); historicalNotices.push(historical);
		payloads.push(historical);
		const calls = requests === 1 && !options.error;
		const msg: any = { role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: 100,
			content: calls ? Array.from({ length: options.parallel ? 2 : 1 }, (_, i) => ({ type: "toolCall", id: `c${i}`, name: "diagnostic", arguments: {} })) : [{ type: "text", text: "DONE" }],
			usage: options.error ? { ...usage, input: 9, cacheRead: -1, cacheWrite: undefined, output: 0 } : { ...usage }, stopReason: options.error ? "aborted" : calls ? "toolUse" : "stop" };
		const stream = createAssistantMessageEventStream(); stream.push(options.error ? { type: "error", reason: "aborted", error: msg } : { type: "done", reason: msg.stopReason, message: msg }); return stream;
	}) as typeof session.agent.streamFunction;
	await session.bindExtensions({});
	return { session, extensionsResult, services, diagnostics: [] };
	}, { cwd: home, agentDir: home, sessionManager: SessionManager.create(home, join(home, "sessions")) });
	return { host, get session() { return host.session; }, payloads, systemNotices, historicalNotices, compactions, starts, ends, toolEnds, handlers, contexts, get requests() { return requests; }, get effects() { return effects; }, async close() { await host.dispose(); await rm(home, { recursive: true, force: true }); } };
}

test("usage-real-sdk-next-payload", async () => {
	for (const compact of [false, true]) { const f = await fixture({ compact }); try {
		await f.session.prompt("inert"); assert.equal(f.requests, 2); assert.equal(f.effects, 1);
		assert.notEqual(f.starts[0], f.ends[0]); const n = f.payloads[1].at(-1);
		assert.equal(n.completedMainRequests, 1); assert.equal(n.knownInclusiveInput, 21); assert.equal(n.knownOutput, 5); assert.equal(n.pendingMainRequests, 0);
	} finally { await f.close(); } }
});

test("usage-identical-values-and-duplicate-event", async () => {
	const f = await fixture(); try { await f.session.prompt("one");
		// Artificial duplicate final event: negative control, not a genuine request.
		await f.handlers.message_end({ message: f.ends[1] }, f.contexts[1]);
		await f.session.prompt("two"); const n = f.systemNotices[2].at(-1);
		assert.equal(n.completedMainRequests, 2); assert.equal(n.knownInclusiveInput, 42); assert.equal(n.knownOutput, 10);
	} finally { await f.close(); }
});

test("usage-unknown-abort-lowerbounds", async () => {
	const f = await fixture({ error: true }); try { await f.session.prompt("abort"); await f.session.prompt("next");
		const n = f.systemNotices[1].at(-1); assert.equal(n.unknownUsageRequests, 1); assert.equal(n.knownInclusiveInput, 9); assert.equal(n.knownOutput, 0);
		assert.equal(n.remainingInputUpperBound, 991); assert.equal(n.reserveState, "unknown");
	} finally { await f.close(); }
});

test("usage-parallel-forgery-preservation", async () => {
	const f = await fixture({ parallel: true, compact: true }); try { await f.session.prompt("parallel"); assert.equal(f.effects, 2);
		assert.equal(f.toolEnds.length, 2); for (const e of f.toolEnds) {
			assert.deepEqual(e.result.details, { distinct: "DETAIL" }); assert.deepEqual(e.result.structuredContent, { distinct: "STRUCTURED" });
			const text = textOf(e.result); assert.ok(text.includes("BODY")); const n = notes(text); assert.equal(n.length, 1); assert.equal(n[0].knownInclusiveInput, 21); assert.ok(!("spoof" in n[0]));
			const t = JSON.parse(text.split("\n").find((l: string) => l.startsWith("ROSCLAW_TOOL_BUDGET_COMPACT_JSON:"))!.split("JSON:")[1]); assert.equal(t.usedTotal, 2);
		}
	} finally { await f.close(); }
});

test("usage-runtime-session-generation", async () => {
	const f = await fixture(); const other = await fixture(); try { await f.session.prompt("one"); const old = f.systemNotices[0].at(-1);
		const saved = f.session.sessionFile!;
		assert.equal((await f.host.newSession()).cancelled, false);
		// Artificial late replay of old settled final is ignored, no historical scanning.
		await f.handlers.message_end({ message: f.ends[0] }, f.contexts[0]);
		await f.session.prompt("new"); const n = f.systemNotices[2].at(-1); assert.equal(n.runtimeGeneration, old.runtimeGeneration); assert.equal(n.completedMainRequests, 2);
		assert.equal((await f.host.switchSession(saved)).cancelled, false);
		await f.session.prompt("resume"); assert.equal(f.systemNotices[3].at(-1).completedMainRequests, 3);
		const entry = f.session.sessionManager.getBranch().find(e => e.type === "message" && e.message.role === "user")!;
		assert.equal((await f.host.fork(entry.id, { position: "at" })).cancelled, false);
		await f.session.prompt("fork"); assert.equal(f.systemNotices[4].at(-1).completedMainRequests, 4);
		await other.session.prompt("other"); assert.notEqual(other.systemNotices[0].at(-1).runtimeGeneration, old.runtimeGeneration); assert.equal(other.systemNotices[0].at(-1).completedMainRequests, 0);
	} finally { await f.close(); await other.close(); }
});

test("usage-compaction-unknown", async () => {
	const f = await fixture({ compaction: true }); try { await f.session.prompt("one");
		// Two real persisted turns and a small keep window make preparation possible.
		await f.session.prompt("history ".repeat(512));
		// Actual manual entry reaches the installed hook and is cancelled offline.
		await assert.rejects(f.session.compact(), /^Error: Compaction cancelled$/);
		assert.equal(f.compactions.length, 1); assert.equal(f.compactions[0].reason, "manual");
		assert.ok(f.compactions[0].preparation); assert.ok(f.compactions[0].branchEntries.length > 0);
		await f.session.prompt("next"); const n = f.systemNotices[3].at(-1); assert.equal(n.unaccountedCompaction, true); assert.equal(n.reserveState, "unknown"); assert.equal(f.requests, 4);
	} finally { await f.close(); }
});

test("usage-three-parser-cli-prehome", async () => {
	const home = await mkdtemp(join(tmpdir(), "usage-cli-")); try {
		validateModelUsageAwareness(conf); createToolCallBudgetExtension({ allowedTools: [], modelUsageAwareness: conf });
		const invalid: unknown[] = [null, [], {}, { ...conf, extra: 1 }, { ...conf, version: true }, { ...conf, lifetime: "session" }];
		for (const key of ["inclusiveInputLimit", "outputLimit", "deliveryReserveInput", "deliveryReserveOutput"]) for (const value of [true, -1, 0.5, 9007199254740992]) invalid.push({ ...conf, [key]: value });
		invalid.push({ ...conf, deliveryReserveInput: 1001 }, { ...conf, outputLimit: 0 });
		for (const value of invalid) {
			assert.throws(() => validateModelUsageAwareness(value)); assert.throws(() => createToolCallBudgetExtension({ allowedTools: [], modelUsageAwareness: value as any }));
			const file = join(home, "policy.json"); await writeFile(file, JSON.stringify({ allowedTools: [], modelUsageAwareness: value }));
			const result = spawnSync(process.execPath, [resolve("dist/src/main.js"), "--tool-call-policy", file], { encoding: "utf8", timeout: 3000, env: { ...process.env, ROSCLAW_HOME: join(home, "unused") } });
			assert.notEqual(result.status, 0); assert.match(result.stderr, /INVALID_TOOL_CALL_POLICY/);
		}
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("usage-reserve-default-compatibility", async () => {
	const plain = await fixture({ enabled: false }); const enabled = await fixture({ reserve: true }); try {
		await plain.session.prompt("plain"); await enabled.session.prompt("enabled"); assert.equal(plain.requests, enabled.requests); assert.equal(plain.effects, enabled.effects);
		assert.ok(plain.systemNotices.every(p => p.length === 0));
		assert.deepEqual(plain.historicalNotices, [[], [{ spoof: true }]]);
		assert.equal(enabled.systemNotices[0].at(-1).reserveState, "finish_delivery");
		assert.equal(enabled.payloads[1].at(-1).completedMainRequests, 1);
		for (const f of [plain, enabled]) {
			assert.deepEqual(f.toolEnds[0].result.details, { distinct: "DETAIL" });
			assert.deepEqual(f.toolEnds[0].result.structuredContent, { distinct: "STRUCTURED" });
		}
		assert.deepEqual(notes(textOf(plain.toolEnds[0].result)), [{ spoof: true }]);
		assert.ok(notes(textOf(enabled.toolEnds[0].result)).every(n => !("spoof" in n)));
		const toolText = textOf(enabled.toolEnds[0].result); const plainText = textOf(plain.toolEnds[0].result);
		const marker = "ROSCLAW_TOOL_POLICY_JSON:";
		assert.equal(toolText.split("\n").find((l: string) => l.startsWith(marker)), plainText.split("\n").find((l: string) => l.startsWith(marker)));
	} finally { await plain.close(); await enabled.close(); }
});

// Managed idle entry: real SDK turns, no manually invoked positive callbacks.
async function managedFixture(enabled = true, partial = false) {
	const home = await mkdtemp(join(tmpdir(), "managed-usage-sdk-"));
	const settings = SettingsManager.inMemory({ compaction: { enabled: false, keepRecentTokens: 1 }, retry: { enabled: false } });
	const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null, modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
	const model: Model<"openai-completions"> = { id: "managed", name: "managed", api: "openai-completions", provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 100 };
	runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT", models: [model] });
	let beforeStarts = 0;
	const compactions: any[] = [], payloads: any[][] = [];
	const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings, noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
		extensionFactories: [{ name: "usage", factory: createToolCallBudgetExtension({ allowedTools: [], ...(enabled ? { modelUsageAwareness: conf } : {}) }) }, { name: "observe", factory: pi => {
			pi.on("before_agent_start", () => { beforeStarts++; });
			pi.on("session_before_compact", e => { compactions.push(e); return { cancel: true }; });
		} }] });
	await loader.reload();
	const { session } = await createAgentSession({ cwd: home, agentDir: home, model: runtime.getModel(model.provider, model.id)!, modelRuntime: runtime, sessionManager: SessionManager.inMemory(home), settingsManager: settings, resourceLoader: loader, tools: [], thinkingLevel: "off" });
	let requests = 0;
	session.agent.streamFunction = ((_m: unknown, context: any) => {
		assert.ok(++requests <= 8);
		payloads.push(structuredClone(context.messages));
		const msg: any = { role: "assistant", api: model.api, provider: model.provider, model: model.id, timestamp: 100, content: [{ type: "text", text: "DONE" }], usage: partial && requests === 1 ? { ...usage, input: 9, cacheRead: -1, cacheWrite: undefined, output: 0 } : { ...usage }, stopReason: partial && requests === 1 ? "aborted" : "stop" };
		const stream = createAssistantMessageEventStream();
		stream.push(partial && requests === 1 ? { type: "error", reason: "aborted", error: msg } : { type: "done", reason: "stop", message: msg });
		return stream;
	}) as typeof session.agent.streamFunction;
	await session.bindExtensions({});
	return { session, payloads, compactions, get beforeStarts() { return beforeStarts; }, get requests() { return requests; }, async close() { session.dispose(); await rm(home, { recursive: true, force: true }); } };
}

async function managedTurn(f: Awaited<ReturnType<typeof managedFixture>>, state: string) {
	assert.ok(f.session.isIdle);
	const content = `后台 Operation owned-${state} 已终止：${state}。用 process_output 查看输出，然后在同一任务里继续（验证/修复/交付）。`;
	const details = { operation_id: `owned-${state}`, state };
	await f.session.sendCustomMessage({ customType: "rosclaw.operation.result", content, display: false, details }, { triggerTurn: true });
	assert.ok(f.session.isIdle);
	const custom = f.payloads.at(-1)!.find(m => m.role === "custom" && m.details?.operation_id === details.operation_id);
	// The public stream converter can render custom messages as user text;
	// persisted SDK history is authoritative for custom type and binding.
	const persisted = f.session.messages.find((m: any) => m.role === "custom" && m.details?.operation_id === details.operation_id) as any;
	assert.ok(persisted); assert.equal(persisted.content, content); assert.deepEqual(persisted.details, details); assert.equal(persisted.display, false);
	assert.ok(custom || f.payloads.at(-1)!.some(m => textOf(m).includes(content)));
	return f.payloads.at(-1)!;
}

function leadingUsage(messages: any[]) {
	assert.equal(messages[0].role, "system");
	assert.equal(messages.filter(m => m.role === "system").length, 1);
	const n = notes(textOf(messages[0]));
	assert.equal(n.length, 1, "ONE_CURRENT_LEADING_USAGE_REQUIRED");
	return n[0];
}

for (const [state, name] of [["SUCCEEDED", "success"], ["FAILED", "failure"], ["CANCELLED", "cancellation"]]) {
	test(`managed idle terminal ${name} uses fresh leading usage`, async () => {
		const f = await managedFixture(); try {
			await f.session.prompt("manual inert prompt"); assert.ok(f.session.isIdle);
			assert.equal(f.requests, 1); assert.equal(f.beforeStarts, 1);
			const initial = leadingUsage(f.payloads[0]); assert.equal(initial.completedMainRequests, 0);
			const messages = await managedTurn(f, state);
			assert.equal(f.beforeStarts, 1, "idle triggerTurn bypasses before_agent_start");
			const n = leadingUsage(messages);
			assert.equal(n.completedMainRequests, 1); assert.equal(n.knownInclusiveInput, 21); assert.equal(n.knownOutput, 5); assert.equal(n.pendingMainRequests, 0);
			assert.equal(n.runtimeGeneration, initial.runtimeGeneration); assert.ok(n.sequence > initial.sequence);
			await f.session.prompt("manual after idle");
			assert.equal(leadingUsage(f.payloads[2]).completedMainRequests, 2);
			assert.equal(f.beforeStarts, 2);
		} finally { await f.close(); }
	});
}

test("managed wakeup preserves unknown and compaction lowerbounds", async () => {
	const f = await managedFixture(true, true); try {
		await f.session.prompt("partial abort");
		const n = leadingUsage(await managedTurn(f, "FAILED"));
		assert.equal(n.completedMainRequests, 1); assert.equal(n.knownInclusiveInput, 9); assert.equal(n.knownOutput, 0); assert.equal(n.unknownUsageRequests, 1); assert.equal(n.reserveState, "unknown");
		await f.session.prompt("history ".repeat(512));
		await assert.rejects(f.session.compact(), /^Error: Compaction cancelled$/);
		assert.equal(f.compactions.length, 1); assert.equal(f.compactions[0].reason, "manual"); assert.ok(f.compactions[0].preparation);
		const after = leadingUsage(await managedTurn(f, "SUCCEEDED"));
		assert.equal(after.completedMainRequests, 3); assert.equal(after.knownInclusiveInput, 51); assert.equal(after.knownOutput, 10); assert.equal(after.unknownUsageRequests, 1); assert.equal(after.unaccountedCompaction, true); assert.equal(after.reserveState, "unknown");
	} finally { await f.close(); }
});

test("managed wakeup preserves body and default-disabled compatibility", async () => {
	const plain = await managedFixture(false); const enabled = await managedFixture(); try {
		for (const f of [plain, enabled]) { await f.session.prompt("manual"); await managedTurn(f, "CANCELLED"); assert.equal(f.requests, 2); assert.equal(f.beforeStarts, 1); }
		assert.ok(plain.payloads.every(p => p.flatMap(m => notes(textOf(m))).length === 0));
		assert.equal(leadingUsage(enabled.payloads[1]).completedMainRequests, 1);
		// Independent sessions have independent wall-clock timestamps. Compare
		// every other public payload field, retaining assistant measured usage.
		const withoutSystems = (p: any[]) => p.filter(m => m.role !== "system").map(({ timestamp: _timestamp, ...message }) => message);
		assert.deepEqual(withoutSystems(enabled.payloads[1]), withoutSystems(plain.payloads[1]));
	} finally { await plain.close(); await enabled.close(); }
});
