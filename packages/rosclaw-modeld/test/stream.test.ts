import assert from "node:assert/strict";
import test from "node:test";
import { toPiMessages } from "../src/stream.js";

test("user and tool result round-trip", () => {
	const messages = toPiMessages([
		{ role: "user", content: "你好" },
		{
			role: "assistant",
			content: null,
			tool_calls: [
				{ id: "c1", type: "function", function: { name: "sim_get_state", arguments: '{"verbose":true}' } },
			],
		},
		{ role: "tool", tool_call_id: "c1", content: '{"ok":true}' },
	]);
	assert.equal(messages.length, 3);
	assert.equal(messages[0].role, "user");
	const assistant = messages[1];
	assert.equal(assistant.role, "assistant");
	if (assistant.role === "assistant") {
		const call = assistant.content[0];
		assert.equal(call.type, "toolCall");
		if (call.type === "toolCall") {
			assert.equal(call.name, "sim_get_state");
			assert.deepEqual(call.arguments, { verbose: true });
		}
	}
	const result = messages[2];
	assert.equal(result.role, "toolResult");
	if (result.role === "toolResult") {
		assert.equal(result.toolCallId, "c1");
	}
});

test("malformed tool arguments degrade to empty object, never throw", () => {
	const messages = toPiMessages([
		{
			role: "assistant",
			content: null,
			tool_calls: [{ id: "c1", function: { name: "t", arguments: "not json" } }],
		},
	]);
	const assistant = messages[0];
	if (assistant.role === "assistant") {
		const call = assistant.content[0];
		if (call.type === "toolCall") assert.deepEqual(call.arguments, {});
	}
});
