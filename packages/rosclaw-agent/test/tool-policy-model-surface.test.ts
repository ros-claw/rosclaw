import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import ts from "typescript";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createRosclawRuntime } from "../src/harness/pi/pi-runtime.js";
import { createRosclawExtension } from "../src/extension/index.js";
import { ActiveSessionContext } from "../src/session/active-context.js";
import { ProductStateCenter } from "../src/session/state-center.js";
import { LocaleManager } from "../src/i18n/locale.js";
import { resolveTaskContext } from "../src/native/active-task-context.js";
import { MODEL_TOOL_NAMES, modelVisibleToolNames, registrationToolNames } from "../src/tools/surface.js";
import { AgentSession, DefaultResourceLoader, SessionManager, SettingsManager, type ModelRuntime, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Agent } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, type AssistantMessage, type Model } from "@earendil-works/pi-ai";

async function createOriginalSdkPreflightFixture(onSettled?: (session: AgentSession) => Promise<void>) {
	const home = await mkdtemp(join(tmpdir(), "rosclaw-original-sdk-edge-"));
	const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
	const model: Model<"openai-completions"> = { id: "localstream-edge", name: "Localstream edge", api: "openai-completions",
		provider: "openai", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 128 };
	let session!: AgentSession;
	let calls = 0;
	const errors: unknown[] = [];
	const events: string[] = [];
	try {
		const factory: ExtensionFactory = pi => {
			if (onSettled) pi.on("agent_settled", async () => { await onSettled(session); });
		};
		const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager,
			noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
			systemPromptOverride: () => "", extensionFactories: [{ name: "localstream-edge", factory }] });
		await loader.reload();
		assert.deepEqual(loader.getExtensions().errors, []);
		const agent = new Agent({ initialState: { systemPrompt: "", model, thinkingLevel: "off", tools: [], messages: [] },
			convertToLlm: messages => messages as never,
			streamFn: () => {
				calls++;
				const stream = createAssistantMessageEventStream();
				const message: AssistantMessage = { role: "assistant", content: [{ type: "text", text: "local" }],
					api: model.api, provider: model.provider, model: model.id, stopReason: "stop", timestamp: Date.now(),
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
				stream.push({ type: "done", reason: "stop", message });
				return stream;
			} });
		// Inert structural auth: the constructor stream is the only stream path.
		const modelRuntime = { hasConfiguredAuth: () => true, checkAuth: async () => ({ type: "api_key" }),
			getModel: () => model, getPhysicalModel: () => model } as unknown as ModelRuntime;
		session = new AgentSession({ agent, sessionManager: SessionManager.inMemory(home), settingsManager,
			cwd: home, resourceLoader: loader, modelRuntime, initialActiveToolNames: [], baseToolsOverride: {} });
		session.subscribe(event => events.push(event.type));
		await session.bindExtensions({ mode: "print", onError: error => errors.push(error) });
		assert.equal(session.prompt, AgentSession.prototype.prompt);
		assert.equal(session.abort, AgentSession.prototype.abort);
		const source = await readFile(new URL("../../src/harness/pi/pi-runtime.ts", import.meta.url), "utf8");
		const ast = ts.createSourceFile("runtime.ts", source, ts.ScriptTarget.Latest, true);
		const declaration = ast.statements.find(node => ts.isFunctionDeclaration(node)
			&& node.name?.text === "installPreflightCancellation");
		assert.ok(declaration);
		const js = ts.transpileModule(declaration.getText(ast), {
			compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
		}).outputText;
		const install = new Function(`${js}; return installPreflightCancellation;`)() as (session: AgentSession) => void;
		install(session);
		return { session, calls: () => calls, errors, events, close: async () => {
			await session.abort(); session.dispose(); await rm(home, { recursive: true, force: true });
		} };
	} catch (error) {
		session?.dispose();
		await rm(home, { recursive: true, force: true });
		throw error;
	}
}

test("original SDK callback abort prevents pending dispatch", { timeout: 15_000 }, async () => {
	const fixture = await createOriginalSdkPreflightFixture();
	try {
		const seen: string[] = [];
		let aborted: Promise<void> | undefined;
		await fixture.session.prompt("cancel in callback", { preflightResult: disposition => {
			seen.push(disposition); aborted = fixture.session.abort();
		} });
		await aborted;
		assert.deepEqual(seen, ["started"]);
		assert.equal(fixture.calls(), 0, "callback abort must prevent actual original agent dispatch");
		const foreign = new Error("ROSCLAW_PREFLIGHT_CANCELLED");
		await assert.rejects(fixture.session.prompt("abort then foreign error", { preflightResult: disposition => {
			seen.push(disposition); aborted = fixture.session.abort(); throw foreign;
		} }), error => error === foreign);
		await aborted;
		assert.deepEqual(seen, ["started", "started"]);
		assert.equal(fixture.calls(), 0);
		await fixture.session.prompt("normal future prompt", { preflightResult: disposition => seen.push(disposition) });
		await fixture.session.waitForIdle();
		assert.equal(fixture.calls(), 1);
		assert.deepEqual(seen, ["started", "started", "started"]);
		assert.deepEqual(fixture.errors, []);
	} finally { await fixture.close(); }
});

test("original SDK agent_settled deferred prompt is cancelled before dispatch", { timeout: 15_000 }, async () => {
	let accepted = false;
	const order: string[] = [];
	const dispositions: string[] = [];
	const fixture = await createOriginalSdkPreflightFixture(async session => {
		if (accepted) return;
		accepted = true;
		order.push("real_agent_settled");
		await session.prompt("accepted deferred action", { preflightResult: disposition => dispositions.push(disposition) });
		order.push("deferred_prompt_returned");
		assert.deepEqual(dispositions, [], "real SDK returns before deferred preflight");
		await session.abort();
		order.push("abort_completed_before_deferred_dispatch");
	});
	try {
		await fixture.session.prompt("seed");
		await fixture.session.waitForIdle();
		assert.deepEqual(order, ["real_agent_settled", "deferred_prompt_returned", "abort_completed_before_deferred_dispatch"]);
		assert.ok(fixture.events.includes("agent_settled"));
		assert.equal(fixture.calls(), 1, "only the seed may stream, not the actual deferred continuation");
		assert.equal(dispositions.length, 0);
		assert.deepEqual(fixture.errors, []);
		await fixture.session.prompt("normal after deferred cancellation", { preflightResult: disposition => dispositions.push(disposition) });
		await fixture.session.waitForIdle();
		assert.equal(fixture.calls(), 2);
		assert.deepEqual(dispositions, ["started"]);
		assert.deepEqual(fixture.errors, []);
	} finally { await fixture.close(); }
});

test("extension UI listeners, async probes and timers belong to their session generation", { timeout: 15_000 }, async () => {
	const home = await mkdtemp(join(tmpdir(), "rosclaw-ui-lifecycle-"));
	const cleanups: Array<() => Promise<void>> = [];
	const pending: Array<(result: Record<string, unknown>) => void> = [];
	const settle = async () => { await new Promise<void>(resolve => setImmediate(resolve)); };
	try {
		const active = new ActiveSessionContext({ sessionId: "one", contextRevision: 0, mode: "SIMULATION",
			profile: "developer", contextState: "LOADING", leaseState: "NONE", actionsAllowed: false });
		let kitState = "READY";
		let probes = 0;
		const center = new ProductStateCenter({ rosclawHome: home, active, operatorSocket: "unused",
			productVersion: "test", call: async (_home, method) => {
				if (method === "pi.operator.status") return new Promise(resolve => pending.push(resolve));
				return { ok: true, sim_policy: "ask", robot_kit: { state: kitState }, body_display: "fixture" };
			}, operatorCallFn: async () => { probes++; return { ok: false } as never; } });
		center.bootstrap = async () => undefined;
		center.refreshCapabilities = async () => undefined;
		await center.statusReport();
		await center.probeOperator(true);
		const locale = new LocaleManager(home);
		const listeners = (center as unknown as { listeners: Set<() => void> }).listeners;
		const localeListeners = (locale as unknown as { listeners: Set<() => void> }).listeners;
		const baseline = listeners.size;
		const localeBaseline = localeListeners.size;
		const make = () => {
			const handlers = new Map<string, Array<(event: any, ctx: any) => Promise<unknown>>>();
			const pi = { on: (name: string, fn: (event: any, ctx: any) => Promise<unknown>) => {
				const callbacks = handlers.get(name) ?? []; callbacks.push(fn); handlers.set(name, callbacks);
			}, registerCommand: () => undefined, registerShortcut: () => undefined,
				registerMessageRenderer: () => undefined, registerEntryRenderer: () => undefined,
				registerTool: () => undefined };
			createRosclawExtension({ profile: "developer", version: "test", systemPrompt: "", active,
				coordinator: {} as never, center, locale, rosclawHome: home,
				taskContext: resolveTaskContext({ cwd: home, rosclawHome: home, mode: "SIMULATION" }),
				osIsolationProbe: () => ({ isolationReady: true }) })(pi as never);
			let disposed = false;
			let staleReads = 0;
			let paints = 0;
			let widgets = 0;
			let notices = 0;
			const ui = new Proxy({ notify: () => { notices++; }, setWidget: () => { widgets++; },
				setTitle: () => undefined, setHeader: () => { paints++; }, setFooter: () => { paints++; },
				setWorkingIndicator: () => undefined, setWorkingMessage: () => undefined,
				setHiddenThinkingLabel: () => undefined }, { get(target, key) {
				if (disposed) { staleReads++; throw new Error(`disposed UI getter: ${String(key)}`); }
				return Reflect.get(target, key);
			} });
			const ctx = { hasUI: true, ui, model: undefined, isIdle: () => true };
			const start = async () => {
				// Local UI wiring test; real coordinator/replacement proof is the public Main PTY test.
				for (const fn of handlers.get("session_start")!.slice(0, 2)) await fn({}, ctx);
			};
			const shutdown = async () => {
				if (disposed) return;
				await handlers.get("session_shutdown")![0]({}, ctx);
				disposed = true;
			};
			cleanups.push(shutdown);
			return { start, shutdown, paints: () => paints, widgets: () => widgets,
				notices: () => notices, staleReads: () => staleReads };
		};
		const first = make();
		await first.start();
		const count = listeners.size;
		assert.equal(count, baseline + 3);
		assert.equal(localeListeners.size, localeBaseline + 1);
		await first.start();
		assert.equal(listeners.size, count, "duplicate start must replace, not accumulate listeners");
		assert.equal(localeListeners.size, localeBaseline + 1);
		active.patch({ contextRevision: 1 });
		await settle();
		assert.ok(pending.length > 0, "a real widget status query is still in flight");
		await first.shutdown();
		assert.equal(listeners.size, baseline);
		assert.equal(localeListeners.size, localeBaseline);
		const oldPaints = first.paints();
		const oldWidgets = first.widgets();
		const oldNotices = first.notices();
		const second = make();
		await second.start();
		const before = second.paints();
		active.patch({ sessionId: "two" });
		assert.ok(second.paints() > before, "current chrome follows active context");
		const beforeLocale = second.paints();
		locale.setUiLocale("zh-CN");
		assert.ok(second.paints() > beforeLocale, "current chrome follows locale changes");
		kitState = "BROKEN";
		await center.statusReport();
		assert.ok(second.notices() > 0, "current kit listener remains responsive");
		for (const resolve of pending.splice(0)) resolve({ ok: true, running: false, enrolled: false });
		await settle();
		assert.ok(second.widgets() > 0, "current widget query can still render");
		assert.equal(first.paints(), oldPaints);
		assert.equal(first.widgets(), oldWidgets);
		assert.equal(first.notices(), oldNotices);
		assert.equal(first.staleReads(), 0, "late completions do not even read disposed UI getters");
		await second.shutdown();
		const probesAtShutdown = probes;
		await new Promise(resolve => setTimeout(resolve, 850));
		assert.equal(probes, probesAtShutdown, "both startup probe timers were cancelled");
		active.patch({ contextRevision: 2 });
		locale.setUiLocale("en-US");
		assert.equal(listeners.size, baseline);
		assert.equal(localeListeners.size, localeBaseline);
		assert.equal(first.staleReads() + second.staleReads(), 0);
	} finally {
		for (const cleanup of cleanups) await cleanup();
		for (const resolve of pending.splice(0)) resolve({ ok: true, running: false });
		await settle();
		await rm(home, { recursive: true, force: true });
	}
});

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: unknown) => void;
	const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
	return { promise, resolve, reject };
}

// Execute the nonexported production installer without adding a product API or
// touching SDK internals. This is an adapter unit test, not an SDK replacement
// or Main acceptance fixture. Actual SDK/Main coverage remains separate.
test("preflight identities preserve callbacks/errors and leave live abort to the SDK", async () => {
	const source = await readFile(new URL("../../src/harness/pi/pi-runtime.ts", import.meta.url), "utf8");
	const ast = ts.createSourceFile("runtime.ts", source, ts.ScriptTarget.Latest, true);
	const declaration = ast.statements.find(node => ts.isFunctionDeclaration(node)
		&& node.name?.text === "installPreflightCancellation");
	assert.ok(declaration);
	const js = ts.transpileModule(declaration.getText(ast), {
		compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
	}).outputText;
	const install = new Function(`${js}; return installPreflightCancellation;`)() as (session: any) => void;
	type Options = { preflightResult?: (disposition: "started" | "handled" | "queued") => void;
		images?: unknown[]; source?: string; streamingBehavior?: string; expandPromptTemplates?: boolean };
	const gates = new Map<string, ReturnType<typeof deferred<void>>>();
	const runs: string[] = [];
	let aborts = 0;
	let running = deferred<void>();
	const session = {
		async prompt(text: string, options?: Options) {
			assert.equal(this, session);
			if (text.startsWith("pending")) await gates.get(text)!.promise;
			if (text === "sdk-error") throw foreign;
			options?.preflightResult?.(text === "handled" ? "handled" : text === "queued" ? "queued" : "started");
			if (text !== "handled" && text !== "queued") runs.push(text);
			if (text === "in-run") await running.promise;
		},
		async abort() { assert.equal(this, session); aborts++; running.resolve(); },
	};
	const foreign = new Error("ROSCLAW_PREFLIGHT_CANCELLED");
	install(session);
	for (const disposition of ["handled", "queued", "started"] as const) {
		const seen: string[] = [];
		await session.prompt(disposition, { preflightResult: value => seen.push(value) });
		assert.deepEqual(seen, [disposition]);
		await assert.rejects(session.prompt(disposition, { preflightResult: () => { throw foreign; } }), error => error === foreign);
	}
	await assert.rejects(session.prompt("sdk-error"), error => error === foreign);
	for (const name of ["pending-one", "pending-two"]) gates.set(name, deferred<void>());
	const one = session.prompt("pending-one");
	const two = session.prompt("pending-two");
	await session.abort();
	await session.prompt("fresh-after-cancel");
	for (const gate of gates.values()) gate.resolve();
	await Promise.all([one, two]);
	assert.ok(!runs.includes("pending-one") && !runs.includes("pending-two"));
	assert.ok(runs.includes("fresh-after-cancel"));
	// Aborting an already started run must delegate, not classify it as preflight.
	running = deferred<void>();
	const started = deferred<void>();
	const live = session.prompt("in-run", { preflightResult: () => started.resolve() });
	await started.promise;
	await session.abort();
	await live;
	assert.equal(aborts, 2);
	await session.prompt("later-live");
	assert.ok(runs.includes("later-live"));
});

for (const stage of ["capability", "context"] as const) {
	test(`before_agent_start ignores replaced ${stage} completion without stale getters/state`, async () => {
		const home = await mkdtemp(join(tmpdir(), "rosclaw-preflight-generation-"));
		const gate = deferred<Record<string, unknown>>();
		const entered = deferred<void>();
		let disposed = false;
		let staleReads = 0;
		let contextCalls = 0;
		const active = new ActiveSessionContext({ sessionId: "old", missionId: "mission-old", contextRevision: 0,
			mode: "SIMULATION", profile: "developer", contextState: "LOADING", leaseState: "NONE", actionsAllowed: false });
		const center = new ProductStateCenter({ rosclawHome: home, active, operatorSocket: "unused", productVersion: "test",
			call: async (_home, method) => {
				if (method === "pi.capability.snapshot" && stage === "capability") { entered.resolve(); return gate.promise; }
				if (method === "pi.context") {
					contextCalls++;
					if (stage === "context") { entered.resolve(); return gate.promise; }
				}
				return { ok: false };
			} });
		const handlers = new Map<string, Array<(event: any, ctx: any) => Promise<any>>>();
		const pi = { on: (event: string, handler: (event: any, ctx: any) => Promise<any>) => {
			const list = handlers.get(event) ?? []; list.push(handler); handlers.set(event, list);
		}, registerCommand: () => undefined, registerShortcut: () => undefined, registerTool: () => undefined,
			registerMessageRenderer: () => undefined, registerEntryRenderer: () => undefined };
		const lateSession = { session: { setActiveToolsByName: () => undefined } };
		try {
			createRosclawExtension({ profile: "developer", version: "test", systemPrompt: "base", active, center,
				coordinator: {} as never, locale: new LocaleManager(home), rosclawHome: home, lateSession,
				taskContext: resolveTaskContext({ cwd: home, rosclawHome: home, mode: "SIMULATION" }) })(pi as never);
			const ctx = { hasUI: false, isIdle: () => true, ui: {}, get model() {
				if (disposed) { staleReads++; throw new Error("disposed model getter"); }
				return undefined;
			} };
			const before = handlers.get("before_agent_start")![0];
			const old = before({ systemPrompt: "original" }, ctx);
			await entered.promise;
			await handlers.get("session_shutdown")![0]({}, ctx);
			disposed = true;
			lateSession.session = { setActiveToolsByName: () => undefined };
			active.patch({ sessionId: "new", missionId: undefined, contextState: "LOADING" });
			const replacement = { ...active.current };
			gate.resolve({ ok: false });
			assert.equal(await old, undefined);
			assert.equal(staleReads, 0);
			assert.deepEqual(active.current, replacement, "old context must not mark the replacement stale");
			assert.equal(contextCalls, stage === "capability" ? 0 : 1);
			assert.deepEqual(await before({ systemPrompt: "live" }, { model: undefined }), { systemPrompt: "live" });
		} finally {
			gate.resolve({ ok: false });
			await handlers.get("session_shutdown")?.[0]({}, {});
			await rm(home, { recursive: true, force: true });
		}
	});
}

test("original SDK handled callbacks keep exact errors and remain usable after abort", { timeout: 15_000 }, async () => {
	const home = await mkdtemp(join(tmpdir(), "rosclaw-sdk-preflight-"));
	let assembled: Awaited<ReturnType<typeof createRosclawRuntime>> | undefined;
	try {
		const rosclawHome = join(home, "private");
		assembled = await createRosclawRuntime({ cwd: home, rosclawHome, profile: "developer", version: "test",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }), toolCallBudget: { allowedTools: ["read"] } });
		const session = assembled.runtime.session;
		const seen: string[] = [];
		await session.prompt("/debug", { preflightResult: disposition => seen.push(disposition) });
		assert.deepEqual(seen, ["handled"]);
		const foreign = new Error("ROSCLAW_PREFLIGHT_CANCELLED");
		await assert.rejects(session.prompt("/debug", { preflightResult: () => { throw foreign; } }), error => error === foreign);
		await session.abort();
		await session.prompt("/debug", { preflightResult: disposition => seen.push(disposition) });
		assert.deepEqual(seen, ["handled", "handled"]);
		assert.deepEqual(session.getActiveToolNames(), ["read"]);
		assert.ok(assembled.ownership.has(session.sessionManager.getSessionFile()!));
	} finally {
		assembled?.runtime.session.dispose();
		assembled?.ownership.releaseAll();
		await rm(home, { recursive: true, force: true });
	}
});

test("intersection has legacy defaults, explicit empty and no foreign tools", () => {
	assert.deepEqual(modelVisibleToolNames(MODEL_TOOL_NAMES), MODEL_TOOL_NAMES);
	assert.deepEqual(modelVisibleToolNames(MODEL_TOOL_NAMES, []), []);
	assert.deepEqual(modelVisibleToolNames(["read", "foreign", "dynamic"], ["dynamic", "read", "absent"]), ["read", "dynamic"]);
	assert.deepEqual(modelVisibleToolNames(["read", "foreign"], ["read", "foreign"]), ["read", "foreign"]);
	assert.deepEqual(modelVisibleToolNames(["read"], ["read", "foreign"]), ["read"]);
});

test("registration authorization is independent of initial availability", () => {
	assert.equal(registrationToolNames(), undefined);
	assert.deepEqual(registrationToolNames([]), []);
	const policy = ["read", "future_capability", "foreign"];
	const registration = registrationToolNames(policy);
	policy.push("later_mutation");
	assert.deepEqual(registration, ["read", "future_capability", "foreign"]);
	assert.deepEqual(modelVisibleToolNames(["read"], registration), ["read"]);
	assert.deepEqual(modelVisibleToolNames(["read", "future_capability"], registration), ["read", "future_capability"]);
	assert.deepEqual(modelVisibleToolNames(["read"], registration), ["read"]);
});

test("unrestricted runtime keeps the full static default active", { timeout: 15_000 }, async () => {
	const home = await mkdtemp(join(tmpdir(), "rosclaw-tool-default-"));
	let assembled: Awaited<ReturnType<typeof createRosclawRuntime>> | undefined;
	try {
		const rosclawHome = join(home, "private");
		assembled = await createRosclawRuntime({ cwd: home, rosclawHome, profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }) });
		assert.deepEqual(assembled.runtime.session.getActiveToolNames(), MODEL_TOOL_NAMES);
	} finally {
		assembled?.runtime.session.dispose();
		await rm(home, { recursive: true, force: true });
	}
});

for (const allowedTools of [["read", "bash"], []] as string[][]) {
	test(`runtime session starts with restricted active tools ${JSON.stringify(allowedTools)}`, { timeout: 15_000 }, async () => {
		const home = await mkdtemp(join(tmpdir(), "rosclaw-tool-surface-"));
		let assembled: Awaited<ReturnType<typeof createRosclawRuntime>> | undefined;
		try {
			const rosclawHome = join(home, "private");
			const policy = { allowedTools: [...allowedTools] };
			assembled = await createRosclawRuntime({ cwd: home, rosclawHome, profile: "developer", version: "fixture",
				taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }), toolCallBudget: policy });
			policy.allowedTools.push("rosclaw_status");
			assert.deepEqual(assembled.runtime.session.getActiveToolNames(), allowedTools);
			const denied = await assembled.runtime.session.extensionRunner?.emitToolCall({ type: "tool_call", toolName: "rosclaw_status", toolCallId: "synthetic", input: {} });
			assert.deepEqual(denied, { block: true, reason: "TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED" });
		} finally {
			assembled?.runtime.session.dispose();
			await rm(home, { recursive: true, force: true });
		}
	});
}
