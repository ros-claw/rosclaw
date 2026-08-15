/** 十五审 PR-RF-3：ACP Client Runtime 红测试（真实 stdio JSON-RPC，
 *  非 PTY scraping）——fake ACP server 按协议应答。
 *
 * 红测试先行——模块不存在必须红：
 * 1. initialize 握手 + capabilities 交换；
 * 2. session/new 返回 sessionId；
 * 3. session/prompt 流式 session/update → TaskEvent 映射（§6.1 表）；
 * 4. session/cancel 终止 prompt；
 * 5. Harness 描述符 readiness（二进制缺失 → 诚实 NOT_INSTALLED）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdirSync, writeFileSync, chmodSync } from "node:fs";

import { AcpClient } from "../src/agent_runtime/acp-client.js";
import { probeHarness } from "../src/agent_runtime/registry.js";
import { mapAcpUpdate } from "../src/agent_runtime/events.js";

/** fake ACP server：stdio ndjson JSON-RPC，支持 initialize/session 生命周期。 */
function fakeAcpServer(dir: string): string {
	const script = `
let buffer = "";
const sessions = new Map();
function send(msg) {
	process.stdout.write(JSON.stringify(msg) + "\\n");
}
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
	buffer += chunk;
	const lines = buffer.split("\\n");
	buffer = lines.pop() ?? "";
	for (const line of lines) {
		if (!line.trim()) continue;
		const msg = JSON.parse(line);
		if (msg.method === "initialize") {
			send({ jsonrpc: "2.0", id: msg.id, result: {
				protocolVersion: 1,
				agentCapabilities: { promptCapabilities: {}, loadSession: true },
			}});
		} else if (msg.method === "session/new") {
			const sessionId = "sess_fake_1";
			sessions.set(sessionId, true);
			send({ jsonrpc: "2.0", id: msg.id, result: { sessionId } });
		} else if (msg.method === "session/prompt") {
			const sessionId = msg.params.sessionId;
			// 流式 updates：agent text → tool call → tool done → 终态。
			setTimeout(() => send({ jsonrpc: "2.0", method: "session/update", params: {
				sessionId, update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "分析中…" } },
			}}), 10);
			setTimeout(() => send({ jsonrpc: "2.0", method: "session/update", params: {
				sessionId, update: { sessionUpdate: "tool_call", toolCallId: "tc1", title: "read", status: "pending" },
			}}), 20);
			setTimeout(() => send({ jsonrpc: "2.0", method: "session/update", params: {
				sessionId, update: { sessionUpdate: "tool_call_update", toolCallId: "tc1", status: "completed" },
			}}), 30);
			setTimeout(() => send({ jsonrpc: "2.0", id: msg.id, result: { stopReason: "end_turn" } }), 40);
		} else if (msg.method === "session/cancel") {
			send({ jsonrpc: "2.0", method: "session/update", params: {
				sessionId: msg.params.sessionId,
				update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "" } },
			}});
		}
	}
});
`;
	const path = `${dir}/fake-acp-server.mjs`;
	writeFileSync(path, script);
	chmodSync(path, 0o755);
	return path;
}

function spawnFake(dir: string): ChildProcess {
	return spawn(process.execPath, [fakeAcpServer(dir)], {
		stdio: ["pipe", "pipe", "inherit"],
	});
}

describe("AcpClient（真实 stdio JSON-RPC）", () => {
	it("initialize + session/new + prompt 流式事件 + cancel", async () => {
		const dir = `/tmp/acp-test-${Date.now()}`;
		mkdirSync(dir, { recursive: true });
		const proc = spawnFake(dir);
		const client = new AcpClient(proc);
		const init = await client.initialize();
		assert.equal(init.protocolVersion, 1);
		const session = await client.newSession({ cwd: "/tmp", mcpServers: [] });
		assert.equal(session.sessionId, "sess_fake_1");
		const events: Array<Record<string, unknown>> = [];
		const unsub = client.onSessionUpdate((sessionId, update) => {
			events.push(mapAcpUpdate(sessionId, update));
		});
		const result = await client.prompt(session.sessionId, "fix the tests");
		assert.equal(result.stopReason, "end_turn");
		assert.deepEqual(events.map((e) => e.kind), [
			"worker.message.delta",
			"worker.tool.started",
			"worker.tool.completed",
		]);
		assert.equal(events[0].text, "分析中…");
		unsub();
		await client.cancel(session.sessionId);
		client.dispose();
	});
});

describe("Harness readiness", () => {
	it("二进制缺失 → 诚实 NOT_INSTALLED（不假装 ready）", () => {
		const descriptor = probeHarness({
			id: "claude-local",
			command: "definitely-not-installed-acp-binary-xyz",
		});
		assert.equal(descriptor.readiness.runtime, "not_installed");
		assert.equal(descriptor.ready, false);
	});

	it("存在的命令 → ready", () => {
		const descriptor = probeHarness({ id: "node-echo", command: process.execPath });
		assert.equal(descriptor.ready, true);
	});
});
