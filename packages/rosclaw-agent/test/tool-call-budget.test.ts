import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createAssistantMessageEventStream, InMemoryCredentialStore, Type, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { createToolCallBudgetExtension, type ToolCallBudget } from "../src/harness/pi/tool-call-budget.js";

for (const policy of [
	{ allowedTools: ["read", "read"] }, { allowedTools: [""] },
	{ allowedTools: ["read"], maxCalls: { read: -1 } },
	{ allowedTools: ["read"], maxCalls: { read: 1.5 } },
	{ allowedTools: ["read"], maxCalls: { read: true } },
	{ allowedTools: ["read"], maxCalls: { bash: 1 } },
	{ allowedTools: ["read"], maxTotalCalls: Number.NaN },
	{ allowedTools: ["read"], maxCall: { read: 1 } },
]) {
	test(`invalid caller budget fails before runtime: ${JSON.stringify(policy)}`, () => {
		assert.throws(() => createToolCallBudgetExtension(policy as unknown as ToolCallBudget), /INVALID_TOOL_CALL_BUDGET/);
	});
}

test("PI backend forwards an invalid budget before private runtime setup", async () => {
	const { createPiBackend } = await import("../src/harness/pi/pi-backend.js");
	const { access } = await import("node:fs/promises");
	const home = await mkdtemp(join(tmpdir(), "rosclaw-budget-admission-"));
	const privateHome = join(home, "private_runtime");
	try {
		await assert.rejects(createPiBackend().create({ cwd: home,
			backendOptions: { rosclawHome: privateHome, toolCallBudget: { allowedTools: ["read"], maxTotalCalls: -1 } },
		}), /INVALID_TOOL_CALL_BUDGET_TOTAL/);
		await assert.rejects(access(privateHome));
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("actual ROSClaw runtime installs the admitted public tool hook", { timeout: 10_000 }, async () => {
	const { createRosclawRuntime } = await import("../src/harness/pi/pi-runtime.js");
	const { resolveTaskContext } = await import("../src/native/active-task-context.js");
	const home = await mkdtemp(join(tmpdir(), "rosclaw-budget-runtime-"));
	let assembled: Awaited<ReturnType<typeof createRosclawRuntime>> | undefined;
	try {
		const rosclawHome = join(home, "private_runtime");
		assembled = await createRosclawRuntime({ cwd: home, rosclawHome, profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }),
			toolCallBudget: { allowedTools: ["read"], maxCalls: { read: 1 } },
		});
		const runner = assembled.runtime.session.extensionRunner;
		assert.ok(runner);
		const event = { type: "tool_call" as const, toolName: "read" as const, toolCallId: "fixture", input: { path: "fixture.txt" } };
		assert.equal(await runner.emitToolCall(event), undefined);
		assert.deepEqual(await runner.emitToolCall({ ...event, toolCallId: "fixture_2" }),
			{ block: true, reason: "TOOL_CALL_BUDGET_EXHAUSTED" });
		assert.deepEqual(await runner.emitToolCall({ type: "tool_call", toolName: "bash", toolCallId: "fixture_3", input: { command: "unused" } }),
			{ block: true, reason: "TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED" });
	} finally {
		assembled?.runtime.session.dispose();
		await rm(home, { recursive: true, force: true });
	}
});

for (const mode of ["per_tool", "total", "zero", "failed_body"] as const) {
test(`real public PI budget ${mode} blocks before effects across prompts`, { timeout: 10_000 }, async () => {
	const home = await mkdtemp(join(tmpdir(), "rosclaw-budget-sdk-"));
	let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
	try {
		const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
		const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
			modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
		const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions",
			provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 100 };
		runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl,
			apiKey: "PRIVATE_OFFLINE_FIXTURE", models: [model] });
		const policy: ToolCallBudget & { allowedTools: string[]; maxCalls: { diagnostic: number } } = {
			allowedTools: ["diagnostic"], maxCalls: { diagnostic: mode === "zero" ? 0 : mode === "total" ? 10 : 1 },
			...(mode === "total" ? { maxTotalCalls: 1 } : {}),
		};
		const factory = createToolCallBudgetExtension(policy);
		// Caller mutation after admission cannot broaden the frozen restriction.
		policy.allowedTools.push("forbidden"); policy.maxCalls.diagnostic = 20;
		const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
			noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
			extensionFactories: [{ name: "actual-budget", factory }] });
		await loader.reload();
		const effects: string[] = [];
		const customTools = ["diagnostic", "forbidden"].map(name => ({ name, label: name, description: "Private harmless counter",
			parameters: Type.Object({}), execute: async () => {
				effects.push(name);
				if (mode === "failed_body") throw new Error("Deliberate harmless fixture failure");
				return { content: [{ type: "text" as const, text: "counter admitted" }], details: {} };
			} }));
		({ session } = await createAgentSession({ cwd: home, agentDir: home, model: runtime.getModel(model.provider, model.id)!,
			modelRuntime: runtime, sessionManager: SessionManager.inMemory(home), settingsManager: settings,
			resourceLoader: loader, tools: ["diagnostic", "forbidden"], customTools, thinkingLevel: "off" }));
		session.setActiveToolsByName(["diagnostic", "forbidden"]);
		let requests = 0;
		session.agent.streamFunction = () => {
			assert.ok(++requests <= 4);
			const names = requests === 1 ? ["diagnostic", "diagnostic", "forbidden"] : requests === 3 ? ["diagnostic"] : [];
			const stream = createAssistantMessageEventStream();
			const message = { role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
				content: names.map((name, i) => ({ type: "toolCall" as const, id: `call_${requests}_${i}`, name, arguments: {} })),
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
				stopReason: names.length ? "toolUse" as const : "stop" as const, timestamp: Date.now() };
			stream.push({ type: "done", reason: message.stopReason, message });
			return stream;
		};
		await session.prompt("offline synthetic tool calls");
		await session.prompt("second offline turn must retain the exhausted budget");
		assert.deepEqual(effects, mode === "zero" ? [] : ["diagnostic"]);
		const results = session.messages.filter(m => m.role === "toolResult");
		assert.equal(results.length, 4);
		assert.equal(results.filter(m => m.isError).length, mode === "zero" || mode === "failed_body" ? 4 : 3);
		assert.ok(session.isIdle);
	} finally {
		session?.dispose();
		await rm(home, { recursive: true, force: true });
	}
});
}

// Declared roles are task policy, not a change to general product role semantics.
type RoleHooks = Record<string, (event: any) => any>;
function roleHooks(extension: ReturnType<typeof createToolCallBudgetExtension>): RoleHooks {
	const hooks: RoleHooks = {};
	extension({ on: (name: string, handler: RoleHooks[string]) => { hooks[name] = handler; } } as never);
	return hooks;
}
function rolePolicy(): ToolCallBudget & { declaredDeliveryRoles: { path: string; role: string }[] } {
	return { allowedTools: ["rosclaw_deliver"], exactPaths: { rosclaw_deliver: ["delivery.json", "other.json"] },
		declaredDeliveryRoles: [{ path: "delivery.json", role: "diagnostic_report" }],
		maxCalls: { rosclaw_deliver: 1 }, maxTotalCalls: 1 };
}
const roleEvent = (input: unknown) => ({ type: "tool_call", toolName: "rosclaw_deliver", toolCallId: "role-fixture", input });
function roleNotice(text: string, compact = false) {
	const marker = compact ? "ROSCLAW_TOOL_BUDGET_COMPACT_JSON:" : "ROSCLAW_TOOL_POLICY_JSON:";
	const lines = text.split("\n").filter(line => line.startsWith(marker));
	assert.equal(lines.length, 1);
	return JSON.parse(lines[0].slice(marker.length));
}

test("declared role hook rejects raw wrong/missing/foreign roles without consuming quota", async () => {
	const home = await mkdtemp(join(tmpdir(), "declared-role-hook-"));
	try {
		const policy = rolePolicy();
		const h = roleHooks(createToolCallBudgetExtension(policy, home));
		policy.declaredDeliveryRoles[0].role = "report";
		policy.declaredDeliveryRoles[0].path = "other.json";
		policy.declaredDeliveryRoles.push({ path: "other.json", role: "report" });
		for (const role of [undefined, null, 1, false, {}, [], "report", "unknown", "Diagnostic_report", " diagnostic_report", "diagnostic_report "]) {
			const denied = await h.tool_call(roleEvent({ path: "delivery.json", ...(role === undefined ? {} : { role }) }));
			assert.match(denied.reason, /^TOOL_CALL_BUDGET_DELIVERY_ROLE_REJECTED\n/);
			const snapshot = roleNotice(denied.reason);
			assert.equal(snapshot.usedTotal, 0);
			assert.deepEqual(snapshot.declaredDeliveryRoles, [{ path: join(home, "delivery.json"), role: "diagnostic_report" }]);
		}
		assert.match((await h.tool_call(roleEvent({ path: "outside.json", role: "diagnostic_report" }))).reason,
			/^TOOL_CALL_BUDGET_EXACT_PATH_REJECTED/);
		assert.equal(await h.tool_call(roleEvent({ path: "./delivery.json", role: "diagnostic_report" })), undefined);
		assert.match((await h.tool_call(roleEvent({ path: "delivery.json", role: "diagnostic_report" }))).reason,
			/^TOOL_CALL_BUDGET_EXHAUSTED/);
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("invalid declared role schemas and canonical bindings fail at construction", async () => {
	const home = await mkdtemp(join(tmpdir(), "declared-role-invalid-"));
	try {
		const invalid: unknown[] = [undefined, null, {}, [], "diagnostic_report", Array(129).fill({ path: "delivery.json", role: "report" }),
			[{ path: "delivery.json" }], [{ role: "report" }], [{ path: "delivery.json", role: "report", predicate: true }],
			[{ path: 1, role: "report" }], [{ path: "../delivery.json", role: "report" }],
			[{ path: "outside.json", role: "report" }], [{ path: home, role: "report" }],
			...[null, 7, {}, "", "unknown", " report", "report ", "REPORT", "diagnostic_", "diagnostic_" + "x".repeat(128)]
				.map(role => [{ path: "delivery.json", role }]),
			[{ path: "delivery.json", role: "report" }, { path: "delivery.json", role: "report" }],
			[{ path: "delivery.json", role: "report" }, { path: "./delivery.json", role: "diagnostic_report" }]];
		for (const declaredDeliveryRoles of invalid) {
			assert.throws(() => createToolCallBudgetExtension({ ...rolePolicy(), declaredDeliveryRoles } as ToolCallBudget, home),
				/INVALID_TOOL_CALL_BUDGET_DELIVERY_ROLE/);
		}
		assert.throws(() => createToolCallBudgetExtension({ ...rolePolicy(), exactPaths: undefined }, home), /INVALID_TOOL_CALL_BUDGET_DELIVERY_ROLES/);
		assert.throws(() => createToolCallBudgetExtension({ ...rolePolicy(), declaredRole: "report" } as ToolCallBudget, home), /INVALID_TOOL_CALL_BUDGET/);
		// Existing symlink/alias canonicalizer is also the declaration binder.
		const { writeFile, symlink } = await import("node:fs/promises");
		await writeFile(join(home, "delivery.json"), "{}");
		await symlink(join(home, "delivery.json"), join(home, "alias.json"));
		assert.throws(() => createToolCallBudgetExtension({ ...rolePolicy(), declaredDeliveryRoles: [
			{ path: "delivery.json", role: "report" }, { path: "alias.json", role: "diagnostic_report" },
		] }, home), /INVALID_TOOL_CALL_BUDGET_DELIVERY_ROLE_PATH/);
		await symlink(tmpdir(), join(home, "escape"));
		assert.throws(() => createToolCallBudgetExtension({ ...rolePolicy(), declaredDeliveryRoles: [
			{ path: "escape/out.json", role: "report" },
		] }, home), /INVALID_TOOL_CALL_BUDGET_DELIVERY_ROLE_PATH/);
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("wrapped direct, parallel and nested deliveries reject before registration/finalization", async () => {
	const home = await mkdtemp(join(tmpdir(), "declared-role-effects-"));
	try {
		const extension = createToolCallBudgetExtension({ ...rolePolicy(), maxCalls: { rosclaw_deliver: 2 }, maxTotalCalls: 2 }, home);
		const h = roleHooks(extension);
		let registrations = 0, finalizations = 0;
		const nestedDenials: Promise<void>[] = [];
		const [wrapped] = extension.wrapTools([{ name: "rosclaw_deliver", label: "fixture", description: "inert registration boundary",
			parameters: Type.Object({ path: Type.String(), role: Type.Optional(Type.Any()) }),
			execute: async (_id: string, input: any) => {
				registrations++;
				if (input.role === "report") finalizations++;
				assert.equal(input.path, join(home, "delivery.json"));
				// A nested invocation uses exactly the same admission wrapper.
				nestedDenials.push(assert.rejects(async () => invoke({ path: "delivery.json", role: "report" }), /DELIVERY_ROLE_REJECTED/));
				return { content: [{ type: "text" as const, text: "inert registered" }], details: {} };
			} }]);
		const invoke = (input: any) => wrapped.execute("fixture", input, undefined, undefined, {} as never);
		// Hook success is not authority for subsequently mutated execute args.
		const input = { path: "delivery.json", role: "diagnostic_report" };
		assert.equal(await h.tool_call(roleEvent(input)), undefined);
		input.role = "report";
		await assert.rejects(async () => invoke(input), /DELIVERY_ROLE_REJECTED/);
		assert.equal(registrations, 0); assert.equal(finalizations, 0);
		await assert.rejects(async () => invoke({ path: "other-place.json", role: "diagnostic_report" }), /EXACT_PATH_REJECTED/);
		const parallel = await Promise.allSettled(Array.from({ length: 8 }, async (_, i) =>
			invoke({ path: "delivery.json", role: i % 2 ? "report" : "diagnostic_report" })));
		await Promise.all(nestedDenials);
		assert.equal(parallel.filter(r => r.status === "fulfilled").length, 2);
		assert.equal(registrations, 2); assert.equal(finalizations, 0);
		await assert.rejects(async () => invoke({ path: "delivery.json", role: "diagnostic_report" }), /EXHAUSTED/);
	} finally { await rm(home, { recursive: true, force: true }); }
});

for (const mode of ["full", "compact"] as const) {
	test(`declared role visible before calls and immutable in ${mode} policy projections`, async () => {
		const home = await mkdtemp(join(tmpdir(), "declared-role-visible-"));
		try {
			const policy = { ...rolePolicy(), visibleBudget: false, visibleBudgetMode: mode };
			const h = roleHooks(createToolCallBudgetExtension(policy, home));
			const options = { forceSystemPrompt: "forced fixture" };
			const start = h.before_agent_start({ systemPrompt: "fixture", systemPromptOptions: options });
			const expected = [{ path: join(home, "delivery.json"), role: "diagnostic_report" }];
			assert.deepEqual(roleNotice(start.systemPrompt).declaredDeliveryRoles, expected);
			assert.deepEqual(roleNotice(options.forceSystemPrompt).declaredDeliveryRoles, expected);
			policy.declaredDeliveryRoles[0].role = "report";
			const render = () => h.tool_result({ content: [{ type: "text", text: "body" }] }).content.map((p: any) => p.text).join("\n");
			const first = roleNotice(render(), mode === "compact");
			assert.deepEqual(first.declaredDeliveryRoles, expected);
			first.declaredDeliveryRoles[0].role = "report";
			assert.deepEqual(roleNotice(render(), mode === "compact").declaredDeliveryRoles, expected);
			assert.deepEqual(roleNotice(options.forceSystemPrompt).declaredDeliveryRoles, expected);
			if (mode === "compact") {
				const alternative = roleHooks(createToolCallBudgetExtension({ ...rolePolicy(), visibleBudgetMode: mode,
					declaredDeliveryRoles: [{ path: "delivery.json", role: "report" }] }, home));
				const other = roleNotice(alternative.tool_result({ content: [] }).content[0].text, true);
				assert.notEqual(other.policyDigest, first.policyDigest);
				assert.equal(roleNotice(render(), true).policyDigest, first.policyDigest);
			}
		} finally { await rm(home, { recursive: true, force: true }); }
	});
}

test("absent declarations preserve report and notice-free compatibility; partial declarations do not broaden paths", async () => {
	const home = await mkdtemp(join(tmpdir(), "declared-role-legacy-"));
	try {
		const { declaredDeliveryRoles: _roles, ...legacy } = rolePolicy();
		const h = roleHooks(createToolCallBudgetExtension(legacy, home));
		assert.equal(h.before_agent_start, undefined); assert.equal(h.tool_result, undefined);
		assert.equal(await h.tool_call(roleEvent({ path: "delivery.json", role: "report" })), undefined);
		const partial = roleHooks(createToolCallBudgetExtension(rolePolicy(), home));
		assert.equal(await partial.tool_call(roleEvent({ path: "other.json", role: "report" })), undefined);
		const finalRole = roleHooks(createToolCallBudgetExtension({ ...rolePolicy(), declaredDeliveryRoles: [{ path: "delivery.json", role: "report" }] }, home));
		assert.equal(await finalRole.tool_call(roleEvent({ path: "delivery.json", role: "report" })), undefined);
	} finally { await rm(home, { recursive: true, force: true }); }
});

for (const mode of ["full", "compact"] as const) {
	test(`offline public SDK sees declared ${mode} role before calling and blocks final-role effects`, { timeout: 10_000 }, async () => {
		const home = await mkdtemp(join(tmpdir(), "declared-role-sdk-"));
		let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
		try {
			const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
			const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
				modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
			const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions", provider: "fixture",
				baseUrl: "http://invalid.local", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 100 };
			runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "OFFLINE_FIXTURE", models: [model] });
			const extension = createToolCallBudgetExtension({ ...rolePolicy(), visibleBudgetMode: mode }, home);
			const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
				noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
				extensionFactories: [{ name: "declared-role", factory: extension }] });
			await loader.reload();
			let registrations = 0, finalizations = 0;
			const customTools = extension.wrapTools([{ name: "rosclaw_deliver", label: "fixture", description: "Inert delivery effect counters",
				parameters: Type.Object({ path: Type.String(), role: Type.Optional(Type.Any()) }),
				execute: async (_id: string, input: any) => {
					registrations++;
					if (input.role === "report") finalizations++;
					return { content: [{ type: "text" as const, text: "inert evidence" }], details: {} };
				} }]);
			({ session } = await createAgentSession({ cwd: home, agentDir: home, model: runtime.getModel(model.provider, model.id)!,
				modelRuntime: runtime, sessionManager: SessionManager.inMemory(home), settingsManager: settings, resourceLoader: loader,
				tools: ["rosclaw_deliver"], customTools, thinkingLevel: "off" }));
			session.setActiveToolsByName(["rosclaw_deliver"]);
			let requests = 0;
			session.agent.streamFunction = ((_model: unknown, context: { messages: { role: string; content: unknown }[] }) => {
				assert.ok(++requests <= 4);
				const system = context.messages.find(message => message.role === "system");
				assert.ok(system);
				assert.deepEqual(roleNotice(system.content as string).declaredDeliveryRoles,
					[{ path: join(home, "delivery.json"), role: "diagnostic_report" }]);
				const inputs: import("@earendil-works/pi-ai").ToolCall["arguments"][] = requests === 1
					? [{ path: "delivery.json", role: "report" }, { path: "delivery.json" },
						{ path: "delivery.json", role: 7 }, { path: "delivery.json", role: "diagnostic_report" }]
					: requests === 3 ? [{ path: "delivery.json", role: "diagnostic_report" }] : [];
				const message = { role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
					content: inputs.map((arguments_, i) => ({ type: "toolCall" as const, id: `role_${requests}_${i}`, name: "rosclaw_deliver", arguments: arguments_ })),
					usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
					stopReason: inputs.length ? "toolUse" as const : "stop" as const, timestamp: Date.now() };
				const stream = createAssistantMessageEventStream();
				stream.push({ type: "done", reason: message.stopReason, message });
				return stream;
			}) as unknown as typeof session.agent.streamFunction;
			await session.prompt("Offline role admission fixture");
			await session.prompt("Offline exhausted quota fixture");
			assert.equal(registrations, 1); assert.equal(finalizations, 0);
			const results = session.messages.filter(message => message.role === "toolResult");
			assert.equal(results.length, 5);
			assert.equal(results.filter(message => message.isError).length, 4);
			for (const result of results) {
				const text = result.content.filter(part => part.type === "text").map(part => part.text).join("\n");
				assert.deepEqual(roleNotice(text, mode === "compact").declaredDeliveryRoles,
					[{ path: join(home, "delivery.json"), role: "diagnostic_report" }]);
			}
			assert.ok(session.isIdle);
		} finally { session?.dispose(); await rm(home, { recursive: true, force: true }); }
	});
}
