import assert from "node:assert/strict";
import test from "node:test";
import { classifyAssistantFailure, ProviderErrorGate } from "../src/native/model-errors.js";
import { mapPiEvent } from "../src/harness/pi/pi-session-adapter.js";

for (const content of [
	[{ type: "thinking", thinking: "private reasoning must never appear in the error" }],
	[{ type: "text", text: "Partial answer" }],
	[{ type: "toolCall", id: "partial", name: "write", arguments: {} }],
]) {
	test(`output truncation is incomplete for ${content[0].type} content`, () => {
		const message = { role: "assistant", stopReason: "length", content };
		const failure = classifyAssistantFailure(message);
		assert.equal(failure?.code, "MODEL_OUTPUT_LIMIT");
		assert.ok(failure?.taskRecoverable);
		const event = mapPiEvent({ type: "message_end", turnId: "t1", message });
		assert.equal(event?.type, "turn.failed");
		if (event?.type === "turn.failed") {
			assert.equal(event.error.code, "MODEL_OUTPUT_LIMIT");
			assert.equal(event.error.retryable, false);
			assert.doesNotMatch(event.error.message, /private reasoning|Partial answer/);
		}
	});
}

test("truncation card is recoverable and deduplicated without declaring a retry", () => {
	const gate = new ProviderErrorGate();
	const error = classifyAssistantFailure({ role: "assistant", stopReason: "length" })!;
	const card = gate.onError(error, { hasActiveTask: true, raw: "stopReason=length" });
	assert.match(card.cardText, /输出上限.*回复未完成/);
	assert.match(card.cardText, /缩小本次任务/);
	assert.doesNotMatch(card.cardText, /自动重试|上下文超限|compact/);
	assert.equal(card.activity?.raw, "stopReason=length");
	assert.equal(gate.onError(error, { hasActiveTask: true }).showCard, false);
	gate.onSuccess();
	assert.equal(gate.onError(error, { hasActiveTask: false }).showCard, true);
});

test("successful stop/toolUse and nonassistant messages are unaffected", () => {
	for (const stopReason of ["stop", "toolUse"]) {
		const message = { role: "assistant", stopReason };
		assert.equal(classifyAssistantFailure(message), undefined);
		assert.equal(mapPiEvent({ type: "message_end", message })?.type, "assistant.completed");
	}
	assert.equal(classifyAssistantFailure({ role: "user", stopReason: "length" }), undefined);
	assert.equal(mapPiEvent({ type: "message_end", message: { role: "toolResult", stopReason: "length" } }), undefined);
	assert.equal(classifyAssistantFailure({ role: "assistant", stopReason: "error", errorMessage: "401 unauthorized" })?.code, "MODEL_CREDENTIAL_INVALID");
	assert.equal(mapPiEvent({ type: "message_end", message: { role: "assistant", stopReason: "aborted" } })?.type, "turn.cancelled");
});
