import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runToolCall, runAgentLoop } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

test("native PI end events and persisted session retain separate failed/successful bash results", { timeout: 5000 }, async () => {
	for (const reverseCompletion of [false, true]) {
	const root = await mkdtemp(join(tmpdir(), "rosclaw-pi-envelope-"));
	try {
		const tool = buildWorkspacePackTools({ root, bwrapPath: () => null }).find(t => t.name === "bash")!;
		let releaseBad!: () => void;
		const goodEnded = new Promise<void>(resolve => { releaseBad = resolve; });
		const tools = [{ ...tool, execute: async (id: string, args: unknown, signal?: AbortSignal, update?: Parameters<typeof tool.execute>[3]) => {
			// Gate on the observed good end event, not shell timing or execution mode.
			if (reverseCompletion && id === "bad") await goodEnded;
			return tool.execute(id, args, signal, update, {} as never);
		} }];
		const session = SessionManager.create(root, join(root, "sessions"));
		const endings: { toolCallId: string; isError: boolean }[] = [];
		let turns = 0;
		await runAgentLoop([{ role: "user", content: "fixture", timestamp: 0 }], { messages: [], tools }, {
			model: { id: "fixture", name: "fixture", api: "openai-completions", provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 100 },
			convertToLlm: messages => messages as never,
		}, event => {
			if (event.type === "tool_execution_end") {
				endings.push({ toolCallId: event.toolCallId, isError: event.isError });
				if (event.toolCallId === "good") releaseBad();
			}
			if (event.type === "message_end") session.appendMessage(event.message as Parameters<typeof session.appendMessage>[0]);
		}, undefined, () => {
			const first = turns++ === 0;
			const message: AssistantMessage = { role: "assistant", api: "openai-completions", provider: "openai", model: "gpt-4o-mini", content: first ? [
				{ type: "toolCall", id: "bad", name: "bash", arguments: { command: "exit 7" } },
				{ type: "toolCall", id: "good", name: "bash", arguments: { command: "printf ok" } },
			] : [{ type: "text", text: "fixture done" }], usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: first ? "toolUse" : "stop", timestamp: 0 };
			const stream = createAssistantMessageEventStream();
			stream.push({ type: "done", reason: message.stopReason as "toolUse" | "stop", message });
			return stream;
		});
		assert.equal(endings.length, 2);
		assert.deepEqual(endings.filter(end => end.toolCallId === "bad"), [{ toolCallId: "bad", isError: true }]);
		assert.deepEqual(endings.filter(end => end.toolCallId === "good"), [{ toolCallId: "good", isError: false }]);
		assert.equal(endings.some(end => end.toolCallId === "bad" && !end.isError), false);
		assert.equal(endings.some(end => end.toolCallId === "good" && end.isError), false);
		if (reverseCompletion) assert.deepEqual(endings, [{ toolCallId: "good", isError: false }, { toolCallId: "bad", isError: true }]);
		const entries = (await readFile(session.getSessionFile()!, "utf8")).trim().split("\n").map(line => JSON.parse(line));
		const results = entries.filter(entry => entry.message?.role === "toolResult");
		assert.deepEqual(results.map(entry => [entry.message.toolCallId, entry.message.isError, entry.message.details.exitCode]), [["bad", true, 7], ["good", false, 0]]);
	} finally { await rm(root, { recursive: true, force: true }); }
	}
});

test("gated reverse completion retains independent bash ID/error/exit associations", { timeout: 5000 }, async () => {
	const root = await mkdtemp(join(tmpdir(), "rosclaw-reverse-"));
	let releaseBad!: () => void;
	const goodEnded = new Promise<void>(resolve => { releaseBad = resolve; });
	try {
		const tool = buildWorkspacePackTools({ root, bwrapPath: () => null }).find(t => t.name === "bash")!;
		const endings: [string, boolean | undefined, unknown][] = [];
		const results = await Promise.all(["bad", "good"].map(async id => {
			if (id === "bad") await goodEnded;
			const result = await tool.execute(id, { command: id === "bad" ? "exit 7" : "printf ok" }, undefined, undefined, {} as never);
			const row: [string, boolean | undefined, unknown] = [id, result.isError, (result.details as { exitCode: number }).exitCode];
			endings.push(row);
			if (id === "good") releaseBad();
			return row;
		}));
		assert.deepEqual(endings, [["good", false, 0], ["bad", true, 7]]);
		assert.deepEqual(results, [["bad", true, 7], ["good", false, 0]]);
	} finally {
		releaseBad();
		await rm(root, { recursive: true, force: true });
	}
});

for (const [label, command, timeout] of [
	["nonzero", "printf 'failed'; exit 7", undefined],
	["signal", "kill -KILL $$", undefined],
	["timeout", "sleep 60", 0.03],
] as const) {
	test(`bash ${label} reaches PI as tool failure`, async () => {
		const root = await mkdtemp(join(tmpdir(), "rosclaw-envelope-"));
		try {
			const tool = buildWorkspacePackTools({ root, bwrapPath: () => null }).find(t => t.name === "bash")!;
			const result = await runToolCall({ type: "toolCall", id: label, name: "bash", arguments: { command, ...(timeout ? { timeout_sec: timeout } : {}) } }, {
				context: { messages: [] },
				tools: [{ ...tool, execute: (id, args, signal, update) => tool.execute(id, args, signal, update, {} as never) }],
				assistantMessage: { role: "assistant", content: [], api: "openai-responses", provider: "fixture", model: "fixture", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "toolUse", timestamp: 0 },
			});
			assert.equal(result.isError, true, "PI currently classifies failed bash as success");
			assert.equal(result.result.isError, true);
		} finally { await rm(root, { recursive: true, force: true }); }
	});
}

test("parallel bash outcomes stay isolated and successful output is explicit", async () => {
	const roots = await Promise.all([0, 1].map(() => mkdtemp(join(tmpdir(), "rosclaw-parallel-"))));
	try {
		const outputs = await Promise.all(roots.map(async (root, i) => {
			const tool = buildWorkspacePackTools({ root, bwrapPath: () => null, bashProgressIntervalMs: 5 }).find(t => t.name === "bash")!;
			return tool.execute(String(i), { command: `printf 'nonce-${i}'; sleep 0.03; exit ${i}` }, undefined, () => { throw new Error("broken observer"); }, {} as never);
		}));
		assert.equal(outputs[0].isError, false);
		assert.equal(outputs[1].isError, true);
		outputs.forEach((output, i) => {
			assert.match(JSON.stringify(output.content), new RegExp(`nonce-${i}`));
			assert.doesNotMatch(JSON.stringify(output.content), new RegExp(`nonce-${1-i}`));
		});
	} finally { await Promise.all(roots.map(root => rm(root, { recursive: true, force: true }))); }
});

test("aborted bash has machine-readable error even when shell traps TERM", async () => {
	const root = await mkdtemp(join(tmpdir(), "rosclaw-aborted-envelope-"));
	try {
		const tool = buildWorkspacePackTools({ root, bwrapPath: () => null }).find(t => t.name === "bash")!;
		const control = new AbortController();
		const pending = tool.execute("abort", { command: "trap 'exit 0' TERM; sleep 60" }, control.signal, undefined, {} as never);
		setTimeout(() => control.abort(), 30);
		const result = await pending;
		assert.equal(result.isError, true);
		assert.equal((result.details as { aborted?: boolean }).aborted, true);
	} finally { await rm(root, { recursive: true, force: true }); }
});
