import assert from "node:assert/strict";
import test from "node:test";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { mapPiEvent, PiHarnessSession } from "../src/harness/pi/pi-session-adapter.js";

test("Pi tool IDs and assistant terminal states survive the adapter", () => {
	assert.deepEqual(mapPiEvent({ type: "tool_execution_start", toolCallId: "call-1", toolName: "bash", args: {} }),
		{ type: "tool.started", callId: "call-1", tool: "bash", args: {} });
	assert.equal(mapPiEvent({ type: "message_end", message: { role: "user" } }), undefined);
	assert.equal(mapPiEvent({ type: "message_end", message: { role: "toolResult" } }), undefined);
	assert.equal(mapPiEvent({ type: "message_end", message: { role: "assistant", stopReason: "aborted" } })?.type, "turn.cancelled");
	assert.equal(mapPiEvent({ type: "message_end", message: { role: "assistant", stopReason: "error" } })?.type, "turn.failed");
	assert.deepEqual(mapPiEvent({ type: "agent_end" }), { type: "session.idle" });
});
test("adapter uses Pi ModelRuntime and closes a pending event read", async () => {
	const model = { id: "test", provider: "fake" };
	let selected: unknown;
	let unsubscribed = false;
	let aborts = 0;
	const raw = {
		sessionId: "test", modelRuntime: { getModel: () => model },
		setModel: async (m: unknown) => { selected = m; },
		subscribe: () => () => { unsubscribed = true; },
		abort: async () => { aborts++; }, dispose: () => {},
	} as unknown as AgentSession;
	const session = new PiHarnessSession(raw, "/tmp");
	await session.setModel({ provider: "fake", model: "test" });
	assert.equal(selected, model);
	const stream = session.events()[Symbol.asyncIterator]();
	const pending = stream.next();
	await session.close();
	assert.equal((await pending).done, true);
	assert.ok(unsubscribed);
	await session.close();
	assert.equal(aborts, 1);
});
