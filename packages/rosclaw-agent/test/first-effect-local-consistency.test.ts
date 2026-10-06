/** Targeted tests: first-effect local admission consistency (P0-C repair).
 *
 * - process_start forwards the canonical session cwd so the agentd
 *   first-effect admission builds the task workspace in the real session
 *   directory (not the private home/tasks fallback);
 * - an admission rejection (ok:false from the bridge) surfaces as a typed
 *   tool error instead of a silent proceed.
 */

import assert from "node:assert/strict";
import test from "node:test";

import type { BridgeToolContext } from "../src/tools/bridge-tools.js";
import { buildProcessTools } from "../src/tools/process-tools.js";

interface BridgeCall {
	method: string;
	params: Record<string, unknown>;
}

interface FakeCtxOptions {
	sessionCwd?: string;
	bridgeResponse?: Record<string, unknown>;
}

function fakeCtx(calls: BridgeCall[], options: FakeCtxOptions = {}): BridgeToolContext & { sessionCwd?: string } {
	const active = {
		current: {
			sessionId: "pi_fixture",
			missionId: "mis_fixture",
			contextRevision: 7,
			bodyHash: "body_fixture",
			mode: "SIMULATION",
		},
	};
	const center = {
		call: async (method: string, params: Record<string, unknown>) => {
			calls.push({ method, params });
			return options.bridgeResponse ?? { ok: true, result: { ok: true, summary: "started" } };
		},
	};
	return {
		rosclawHome: "/tmp/rosclaw-first-effect-fixture",
		active: active as unknown as BridgeToolContext["active"],
		center: center as unknown as BridgeToolContext["center"],
		...(options.sessionCwd !== undefined ? { sessionCwd: options.sessionCwd } : {}),
	};
}

type StartTool = {
	name: string;
	execute: (id: string, params: { command: string }) => Promise<{
		content: { type: string; text: string }[];
		details: Record<string, unknown>;
		isError: boolean;
	}>;
};

function processStart(tools: ReturnType<typeof buildProcessTools>): StartTool {
	const tool = tools.find((t) => t.name === "process_start");
	assert.ok(tool, "process_start tool must be registered");
	return tool as unknown as StartTool;
}

test("process_start forwards the canonical session cwd for first-effect admission", async () => {
	const calls: BridgeCall[] = [];
	const tools = buildProcessTools(fakeCtx(calls, { sessionCwd: "/canonical/session/workspace" }));
	const result = await processStart(tools).execute("t1", { command: "printf hi" });
	assert.equal(result.isError, false);
	assert.equal(calls.length, 1);
	assert.equal(calls[0].method, "pi.tools.execute");
	const request = (calls[0].params as { request: { tool_name: string; arguments: Record<string, unknown> } }).request;
	assert.equal(request.tool_name, "rosclaw_process_start");
	assert.equal(request.arguments.cwd, "/canonical/session/workspace");
});

test("process_start without a session cwd sends an explicit empty cwd", async () => {
	const calls: BridgeCall[] = [];
	const tools = buildProcessTools(fakeCtx(calls));
	await processStart(tools).execute("t2", { command: "printf hi" });
	const request = (calls[0].params as { request: { arguments: Record<string, unknown> } }).request;
	assert.equal(request.arguments.cwd, "");
});

test("admission rejection from the bridge surfaces as a typed tool error", async () => {
	const calls: BridgeCall[] = [];
	const tools = buildProcessTools(fakeCtx(calls, {
		sessionCwd: "/canonical/session/workspace",
		bridgeResponse: { ok: false, error: "this session does not hold the writer lease", code: "WRITER_LEASE_REQUIRED" },
	}));
	const result = await processStart(tools).execute("t3", { command: "printf hi" });
	assert.equal(result.isError, true);
});
