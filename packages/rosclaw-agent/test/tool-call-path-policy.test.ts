import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Type, InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager,
	type ExtensionAPI, type ToolCallEvent, type ToolCallEventResult, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { createToolCallBudgetExtension, type ToolCallBudget, type ToolCallBudgetExtension } from "../src/harness/pi/tool-call-budget.js";

function fixture(fn: (root: string, outside: string) => unknown) {
	const root = mkdtempSync(join(tmpdir(), "exact-files-"));
	const outside = mkdtempSync(join(tmpdir(), "exact-outside-"));
	writeFileSync(join(root, "allowed.txt"), "allowed");
	writeFileSync(join(root, "protected.txt"), "protected");
	writeFileSync(join(outside, "secret.txt"), "outside");
	symlinkSync(outside, join(root, "escape"));
	return Promise.resolve().then(async () => { await fn(root, outside); }).finally(() => {
		rmSync(root, { recursive: true, force: true }); rmSync(outside, { recursive: true, force: true });
	});
}
function hook(factory: ToolCallBudgetExtension) {
	let call!: (event: ToolCallEvent, ctx: { cwd: string }) => Promise<ToolCallEventResult | undefined>;
	factory({ on(name: string, handler: typeof call) { if (name === "tool_call") call = handler; } } as unknown as ExtensionAPI);
	return (name: string, path: unknown, cwd = "/unrelated") => call({ type: "tool_call", toolName: name,
		toolCallId: "path-fixture", input: { path } } as ToolCallEvent, { cwd });
}
const names = ["read", "write", "edit", "rosclaw_deliver"];

test("exact file binding: abs/relative aliases, immutable authority, denials nonconsuming", () => fixture(async (root, outside) => {
	const policy = { allowedTools: [...names], maxTotalCalls: 4, visibleBudget: true,
		exactPaths: Object.fromEntries(names.map(name => [name, ["allowed.txt"]])) };
	const call = hook(createToolCallBudgetExtension(policy, root));
	policy.exactPaths.write.push("protected.txt"); policy.allowedTools.push("bash"); policy.maxTotalCalls = 100;
	for (const name of names) {
		for (const p of ["protected.txt", "../bad", "a/../allowed.txt", `${root}-sibling/file`, join(outside, "secret.txt"),
			"escape/secret.txt", undefined, null, 42, "", "allowed.txt/child"]) {
			const result = await call(name, p);
			assert.equal(result?.block, true); assert.match(result!.reason!, /TOOL_CALL_BUDGET_EXACT_PATH/);
			const notice = JSON.parse(result!.reason!.split("ROSCLAW_TOOL_POLICY_JSON:")[1]);
			assert.equal(notice.usedTotal, 0); assert.deepEqual(notice.exactPaths.write, [join(root, "allowed.txt")]);
			assert.equal(notice.workspaceRoot, root);
		}
	}
	for (const [i, name] of names.entries()) assert.equal(await call(name, i % 2 ? "allowed.txt" : join(root, "allowed.txt")), undefined);
	assert.match((await call("write", "allowed.txt"))!.reason!, /EXHAUSTED/);
	assert.equal(readFileSync(join(root, "protected.txt"), "utf8"), "protected");
	assert.equal(readFileSync(join(outside, "secret.txt"), "utf8"), "outside");
}));

test("factory rejects malformed schema and requires explicit selected root", () => fixture((root, outside) => {
	for (const exactPaths of [[], null, { write: [] }, { write: [42] }, { write: [""] }, { write: ["../x"] },
		{ write: ["allowed.txt", "allowed.txt"] }, { bash: ["allowed.txt"] }, { read: ["allowed.txt"] }]) {
		assert.throws(() => createToolCallBudgetExtension({ allowedTools: ["write"], exactPaths } as unknown as ToolCallBudget, root), /INVALID_TOOL_CALL_BUDGET/);
	}
	const policy = { allowedTools: ["write"], exactPaths: { write: ["allowed.txt"] } };
	assert.throws(() => createToolCallBudgetExtension(policy), /PATH_WORKSPACE_REQUIRED/);
	assert.throws(() => createToolCallBudgetExtension(policy, "."), /PATH_WORKSPACE_REQUIRED/);
	mkdirSync(join(root, "directory"));
	symlinkSync(join(outside, "secret.txt"), join(root, "leaf"));
	symlinkSync(join(root, "missing"), join(root, "dangling"));
	for (const p of [outside + "/secret.txt", "escape/secret.txt", "leaf", "directory", "dangling"]) {
		assert.throws(() => createToolCallBudgetExtension({ ...policy, exactPaths: { write: [p] } }, root), /EXACT_PATH_BINDING/);
	}
	assert.throws(() => createToolCallBudgetExtension({ ...policy, exactPaths: { write: ["allowed.txt", root + "/allowed.txt"] } }, root), /EXACT_PATH_BINDING/);
	assert.doesNotThrow(() => createToolCallBudgetExtension({ allowedTools: [], exactPaths: {} }, root));
	assert.doesNotThrow(() => createToolCallBudgetExtension({ allowedTools: [], maxTotalCalls: 0 }));
}));

test("nonexistent leaves do not create parents; missing reads remain admitted IO errors", () => fixture(async root => {
	const factory = createToolCallBudgetExtension({ allowedTools: ["write", "read"], maxCalls: { read: 1 },
		exactPaths: { write: ["new/leaf.txt"], read: ["missing.txt"] } }, root);
	const call = hook(factory);
	assert.equal((await call("write", "new/other.txt"))?.block, true);
	assert.equal(existsSync(join(root, "new")), false);
	assert.equal(await call("write", "new/leaf.txt"), undefined);
	assert.equal(await call("read", "missing.txt"), undefined);
	assert.match((await call("read", "missing.txt"))!.reason!, /EXHAUSTED/);
	assert.equal(existsSync(join(root, "new")), false);
}));

test("symlink alias within root binds one file; later retarget cannot broaden", () => fixture(async (root, outside) => {
	symlinkSync(join(root, "allowed.txt"), join(root, "alias"));
	const call = hook(createToolCallBudgetExtension({ allowedTools: ["read"], exactPaths: { read: ["alias"] } }, root));
	assert.equal(await call("read", "allowed.txt"), undefined);
	unlinkSync(join(root, "alias")); symlinkSync(join(outside, "secret.txt"), join(root, "alias"));
	assert.match((await call("read", "alias"))!.reason!, /EXACT_PATH/);
}));

// The SDK's real runner mutates event.input in registration order. Final execute
// admission must not trust the original hook, whether mutation is before or after it.
for (const order of ["before", "after"] as const) {
	test(`actual public PI runner ${order}-budget mutation cannot reach body; recovery shares counters`, { timeout: 10000 }, () => fixture(async root => {
		const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
		const modelRuntime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
			modelsStorePath: join(root, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
		const factory = createToolCallBudgetExtension({ allowedTools: ["write"], maxCalls: { write: 1 },
			exactPaths: { write: ["allowed.txt"] } }, root);
		let mutate = true;
		const mutator = { name: "private-mutator", factory: (pi: ExtensionAPI) => {
			pi.on("tool_call", event => { if (mutate) (event.input as Record<string, unknown>).path = "protected.txt"; });
		} };
		const budget = { name: "budget", factory };
		const loader = new DefaultResourceLoader({ cwd: root, agentDir: root, settingsManager: settings, noExtensions: true,
			noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
			extensionFactories: order === "before" ? [mutator, budget] : [budget, mutator] });
		await loader.reload();
		let effects = 0;
		const tools = factory.wrapTools([{ name: "write", label: "fixture write", description: "Private file byte fixture",
			parameters: Type.Object({ path: Type.String() }), execute: async (_id, params) => {
				effects++; writeFileSync((params as { path: string }).path, "updated");
				return { content: [{ type: "text", text: "written" }], details: {} };
			} } as ToolDefinition<any, any>]);
		const { session } = await createAgentSession({ cwd: root, agentDir: root, modelRuntime,
			sessionManager: SessionManager.inMemory(root), settingsManager: settings, resourceLoader: loader,
			customTools: tools, tools: ["write"] });
		try {
			const runner = session.extensionRunner!;
			const input = { path: "allowed.txt" };
			const denied = await runner.emitToolCall({ type: "tool_call", toolName: "write", toolCallId: "mutated", input });
			if (order === "before") assert.match(denied!.reason!, /EXACT_PATH/);
			else { assert.equal(denied, undefined); assert.throws(() => tools[0].execute("mutated", input, undefined, undefined, {} as never), /EXACT_PATH/); }
			assert.equal(effects, 0); assert.equal(readFileSync(join(root, "protected.txt"), "utf8"), "protected");
			mutate = false;
			const good = { path: join(root, "allowed.txt") };
			assert.equal(await runner.emitToolCall({ type: "tool_call", toolName: "write", toolCallId: "recovery", input: good }), undefined);
			await tools[0].execute("recovery", good, undefined, undefined, {} as never);
			assert.equal(effects, 1); assert.equal(readFileSync(good.path, "utf8"), "updated");
			assert.match((await runner.emitToolCall({ type: "tool_call", toolName: "write", toolCallId: "exhausted", input: good }))!.reason!, /EXHAUSTED/);
		} finally { session.dispose(); }
	}));
}

test("final execution admissions share counts for concurrent/nested calls and failures", () => fixture(async root => {
	const factory = createToolCallBudgetExtension({ allowedTools: ["read"], maxTotalCalls: 1,
		exactPaths: { read: ["missing.txt"] } }, root);
	const tools = factory.wrapTools([{ name: "read", label: "read", description: "Missing-file fixture", parameters: Type.Object({ path: Type.String() }),
		execute: async (_id, p) => { readFileSync((p as { path: string }).path); return { content: [], details: {} }; } }]);
	const call = hook(factory);
	assert.equal(await call("read", "missing.txt"), undefined);
	await assert.rejects(tools[0].execute("failed", { path: "missing.txt" }, undefined, undefined, {} as never), /ENOENT/);
	assert.match((await call("read", "missing.txt"))!.reason!, /EXHAUSTED/);
}));
