import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createAssistantMessageEventStream, InMemoryCredentialStore, Type, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { createToolCallBudgetExtension, type ToolCallBudget } from "../src/harness/pi/tool-call-budget.js";

test("compact enum strict validation and false suppression", () => {
	for (const mode of [null, true, 1, "", "FULL", " compact", [], {}]) {
		assert.throws(() => createToolCallBudgetExtension({ allowedTools: [], visibleBudgetMode: mode } as unknown as ToolCallBudget), /INVALID_TOOL_CALL_BUDGET/);
	}
	const h = hooks({ allowedTools: [], visibleBudget: false, visibleBudgetMode: "compact" });
	assert.equal(h.tool_result, undefined);
	assert.equal(h.before_agent_start, undefined);
});

const MARKER = "ROSCLAW_TOOL_POLICY_JSON:";
const notice = (text: string) => {
	const rows = String(text).split("\n").filter(line => line.startsWith(MARKER));
	assert.equal(rows.length, 1, "exactly one policy marker line");
	return JSON.parse(rows[0].slice(MARKER.length));
};
const expected = (policy: { allowedTools: string[]; exactCommands?: Record<string, string[]>; maxCalls?: Record<string, number>; maxTotalCalls?: number }, used: Record<string, number> = {}) => {
	const usedCalls = Object.fromEntries(policy.allowedTools.map(name => [name, used[name] ?? 0]));
	const usedTotal = Object.values(usedCalls).reduce((a, b) => a + b, 0);
	return {
		allowedTools: [...policy.allowedTools],
		exactCommands: policy.exactCommands ?? {},
		maxCalls: policy.maxCalls ?? {},
		maxTotalCalls: policy.maxTotalCalls ?? null,
		usedCalls,
		usedTotal,
		remainingCalls: Object.fromEntries(policy.allowedTools.map(name => [name,
			policy.maxCalls?.[name] === undefined ? null : Math.max(0, policy.maxCalls[name] - usedCalls[name])])),
		remainingTotal: policy.maxTotalCalls === undefined ? null : Math.max(0, policy.maxTotalCalls - usedTotal),
	};
};
type Handlers = Record<string, (event: never) => unknown>;
const hooks = (policy: ToolCallBudget) => {
	const registered: Handlers = {};
	createToolCallBudgetExtension(policy)({ on: (name: string, cb: never) => {
		assert.equal(registered[name], undefined);
		registered[name] = cb;
		return () => {};
	} } as never);
	return registered;
};
const call = (toolName: string, input: unknown, toolCallId = "fixture") =>
	({ type: "tool_call", toolName, toolCallId, input }) as never;

for (const policy of [
	{ allowedTools: ["d"], exactCommands: { other: ["ok"] } },
	{ allowedTools: ["d"], exactCommands: { d: [] } },
	{ allowedTools: ["d"], exactCommands: { d: ["ok", "ok"] } },
	{ allowedTools: ["d"], exactCommands: { d: [1] } },
	{ allowedTools: ["d"], exactCommands: { d: [""] } },
	{ allowedTools: ["d"], exactCommands: [] },
	{ allowedTools: ["d"], visibleBudget: "true" },
]) {
	test(`invalid exact/visible budget fails before runtime: ${JSON.stringify(policy)}`, () => {
		assert.throws(() => createToolCallBudgetExtension(policy as unknown as ToolCallBudget), /INVALID_TOOL_CALL_BUDGET/);
	});
}

test("exact commands compare raw bytes; rejection consumes no quota; visible snapshot is exact", async () => {
	const policy = { visibleBudget: true, allowedTools: ["d"], exactCommands: { d: ["ok"] }, maxCalls: { d: 1 }, maxTotalCalls: 1 };
	const h = hooks(policy);
	(policy.exactCommands.d as string[]).push("evil");
	(policy.allowedTools as string[]).push("other");
	policy.maxCalls.d = 100;
	for (const input of [{ command: "evil" }, { command: " ok" }, { command: "ok " }, { command: "ok && evil" }, { command: 1 }, {}]) {
		const result = await h.tool_call(call("d", input)) as { block: boolean; reason: string };
		assert.equal(result.block, true);
		assert.match(result.reason, /TOOL_CALL_BUDGET_EXACT_COMMAND_REJECTED/);
		assert.deepEqual(notice(result.reason), expected({ allowedTools: ["d"], exactCommands: { d: ["ok"] }, maxCalls: { d: 1 }, maxTotalCalls: 1 }));
	}
	assert.equal(await h.tool_call(call("d", { command: "ok" }, "one")), undefined);
	const exhausted = await h.tool_call(call("d", { command: "ok" }, "two")) as { block: boolean; reason: string };
	assert.equal(exhausted.block, true);
	assert.match(exhausted.reason, /TOOL_CALL_BUDGET_EXHAUSTED/);
});

test("exact command enforcement works without visibleBudget and keeps old notice-free output", async () => {
	const h = hooks({ allowedTools: ["d"], exactCommands: { d: ["ok"] } });
	assert.deepEqual(await h.tool_call(call("d", { command: "nope" })),
		{ block: true, reason: "TOOL_CALL_BUDGET_EXACT_COMMAND_REJECTED" });
	assert.equal(h.before_agent_start, undefined);
	assert.equal(h.tool_result, undefined);
});

test("parallel admissions of a single-slot budget admit exactly one call", async () => {
	const h = hooks({ visibleBudget: true, allowedTools: ["d"], maxCalls: { d: 1 } });
	const admissions = await Promise.all(Array.from({ length: 12 }, (_, i) => h.tool_call(call("d", {}, `parallel${i}`))));
	assert.equal(admissions.filter(x => x === undefined).length, 1);
});

test("tool_result handler appends one marker, preserves text/details/isError, replaces own prior marker", async () => {
	const h = hooks({ visibleBudget: true, allowedTools: ["d"] });
	const result = await (h.tool_result as (e: unknown) => Promise<{ content: { type: string; text?: string }[]; structuredContent?: unknown }>)({
		type: "tool_result", toolName: "d", toolCallId: "x", input: {},
		content: [{ type: "text", text: `ORIGINAL_BODY_TEXT\n${MARKER}{"stale":true}` }, { type: "image", data: "px" }],
		structuredContent: { keep: true }, isError: false,
	});
	const text = result.content.filter(p => p.type === "text").map(p => p.text).join("\n");
	assert.ok(text.includes("ORIGINAL_BODY_TEXT"));
	assert.deepEqual(notice(text), expected({ allowedTools: ["d"] }));
	assert.deepEqual(result.structuredContent, { keep: true });
});

test("real public PI visible budget decorates system prompt and tool results across prompts", { timeout: 10_000 }, async () => {
	const home = await mkdtemp(join(tmpdir(), "rosclaw-budget-visible-"));
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
		const policy = { visibleBudget: true, allowedTools: ["diagnostic"], exactCommands: { diagnostic: ["good"] },
			maxCalls: { diagnostic: 1 } };
		const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
			noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
			extensionFactories: [{ name: "visible-budget", factory: createToolCallBudgetExtension(policy) }] });
		await loader.reload();
		const effects: unknown[] = [];
		const customTools = ["diagnostic", "forbidden"].map(name => ({ name, label: name, description: "harmless fixture",
			parameters: Type.Object({ command: Type.Optional(Type.Any()) }), execute: async (_id: string, input: unknown) => {
				effects.push(input);
				return { content: [{ type: "text" as const, text: "ORIGINAL_BODY_TEXT" }], details: { preserved: true } };
			} }));
		({ session } = await createAgentSession({ cwd: home, agentDir: home, model: runtime.getModel(model.provider, model.id)!,
			modelRuntime: runtime, sessionManager: SessionManager.inMemory(home), settingsManager: settings,
			resourceLoader: loader, tools: ["diagnostic", "forbidden"], customTools, thinkingLevel: "off" }));
		session.setActiveToolsByName(["diagnostic", "forbidden"]);
		let requests = 0;
		const snapshots: ReturnType<typeof notice>[] = [];
		session.agent.streamFunction = ((_m: unknown, context: { messages: { role: string; content: unknown }[] }) => {
			assert.ok(++requests <= 4);
			// Public PI 1.0.4 normalizes the hook-returned systemPrompt into the
			// leading context.messages system entry; there is no context.systemPrompt.
			const systems = context.messages.filter(x => x.role === "system");
			assert.equal(systems.length, 1, "one leading system entry");
			assert.equal(context.messages[0].role, "system");
			assert.equal(typeof systems[0].content, "string");
			snapshots.push(notice(systems[0].content as string));
			const calls: { name: string; arguments: Record<string, string> }[] = requests === 1
				? [{ name: "diagnostic", arguments: { command: "wrong" } },
					{ name: "diagnostic", arguments: { command: "good" } }, { name: "forbidden", arguments: {} }]
				: [];
			const stream = createAssistantMessageEventStream();
			const message = { role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
				content: calls.map((c, i) => ({ type: "toolCall" as const, id: `call_${requests}_${i}`, ...c })),
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
				stopReason: calls.length ? "toolUse" as const : "stop" as const, timestamp: Date.now() };
			stream.push({ type: "done", reason: message.stopReason, message });
			return stream;
		}) as unknown as typeof session.agent.streamFunction;
		await session.prompt("offline synthetic visible budget batch");
		assert.deepEqual(effects, [{ command: "good" }]);
		assert.deepEqual(snapshots[0], expected(policy));
		const results = session.messages.filter(m => m.role === "toolResult");
		assert.equal(results.length, 3);
		assert.equal(results.filter(m => m.isError).length, 2);
		for (const r of results) {
			const text = (r.content as { type: string; text?: string }[]).filter(x => x.type === "text").map(x => x.text).join("\n");
			const snap = notice(text);
			assert.ok(snap.usedTotal <= 1);
			if (!r.isError) {
				assert.ok(text.includes("ORIGINAL_BODY_TEXT"));
				assert.deepEqual((r as { details?: unknown }).details, { preserved: true });
			}
		}
		assert.ok(session.isIdle);
	} finally {
		session?.dispose();
		await rm(home, { recursive: true, force: true });
	}
});
