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
		sessionId: "test", isIdle: true, modelRuntime: { getModel: () => model },
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

for (const prefix of ["", "auto_"]) {
	test(`PI ${prefix}compaction lifecycle preserves failures and cancellation`, () => {
		assert.deepEqual(mapPiEvent({ type: `${prefix}compaction_start`, reason: "threshold" }), { type: "compaction.started" });
		assert.deepEqual(mapPiEvent({ type: `${prefix}compaction_end`, result: { summary: "retained summary" }, aborted: false }), { type: "compaction.completed" });
		assert.equal(mapPiEvent({ type: `${prefix}compaction_end`, aborted: true })?.type, "compaction.cancelled");
		const failed = mapPiEvent({ type: `${prefix}compaction_end`, errorMessage: "provider unavailable", aborted: false });
		assert.equal(failed?.type, "compaction.failed");
		if (failed?.type === "compaction.failed") assert.equal(failed.error.message, "provider unavailable");
		assert.equal(mapPiEvent({ type: `${prefix}compaction_end`, result: undefined, aborted: false })?.type, "compaction.failed");
	});
}
