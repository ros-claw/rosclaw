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
