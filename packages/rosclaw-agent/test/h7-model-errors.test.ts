/** PR-H7 测试：provider 错误分类（总纲 v2 §8.4 + Gate F 子集）。 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { classifyModelError, ProviderErrorGate } from "../src/native/model-errors.js";

test("H7: 403 配额耗尽 ≠ auth 错误", () => {
	const err = classifyModelError(
		'403: {"message":"You\'ve reached your usage limit for this billing cycle.","type":"access_terminated_error"}',
	);
	assert.equal(err.code, "PROVIDER_QUOTA_EXHAUSTED");
	assert.ok(err.taskRecoverable);
	assert.match(err.recovery, /model/);
});

test("H7: 401 凭据无效", () => {
	assert.equal(classifyModelError("401 unauthorized").code, "MODEL_CREDENTIAL_INVALID");
});

test("H7: 429 限流", () => {
	assert.equal(
		classifyModelError("429: engine overloaded rate limit").code,
		"PROVIDER_RATE_LIMITED",
	);
});

test("H7: 网络不可达", () => {
	assert.equal(classifyModelError("fetch failed ECONNRESET").code, "PROVIDER_UNAVAILABLE");
});

test("H7: 模型不存在", () => {
	assert.equal(
		classifyModelError("model k3-old not found").code,
		"MODEL_NOT_FOUND",
	);
});

test("H7: 上下文超限", () => {
	assert.equal(
		classifyModelError("context length exceeded").code,
		"MODEL_CONTEXT_LIMIT",
	);
});

test("二轮自审：403 并发限额 → PROVIDER_RATE_LIMITED（不是凭据/未分类）", () => {
	const err = classifyModelError(
		'403 {"error":{"type":"permission_error","message":"You\'ve reached your concurrent request limit"}}',
	);
	assert.equal(err.code, "PROVIDER_RATE_LIMITED", `误分类: ${err.code}`);
	assert.match(err.recovery, /重试|换模型/);
});


test("timeouts and aborts are classified as recoverable", () => {
	assert.equal(classifyModelError("Request timed out.").code, "PROVIDER_UNAVAILABLE");
	assert.equal(classifyModelError("Operation aborted").code, "MODEL_REQUEST_CANCELLED");
	assert.equal(classifyModelError("This operation was aborted").code, "MODEL_REQUEST_CANCELLED");
	assert.equal(classifyModelError("The request was canceled").code, "MODEL_REQUEST_CANCELLED");
});

test("user cancellation clears provider pause without suggesting a model switch", () => {
	const gate = new ProviderErrorGate();
	gate.onError(classifyModelError("fetch failed"), { hasActiveTask: true });
	assert.equal(gate.pausedCode, "PROVIDER_UNAVAILABLE");
	const result = gate.onError(classifyModelError("This operation was aborted"), {
		hasActiveTask: true,
		raw: "This operation was aborted",
	});
	assert.equal(gate.pausedCode, null);
	assert.match(result.cardText, /同一任务/);
	assert.doesNotMatch(result.cardText, /model|模型调用失败|配额/);
	assert.equal(result.activity?.code, "MODEL_REQUEST_CANCELLED");
});

test("live OpenAI websocket restart is recoverable provider disconnect, not unknown", () => {
	for (const raw of ["WebSocket closed 1012", "WebSocket closed 1006", "socket hang up"]) {
		const error = classifyModelError(raw);
		assert.equal(error.code, "PROVIDER_UNAVAILABLE");
		assert.ok(error.taskRecoverable);
		assert.match(error.recovery, /重新发送/);
		assert.doesNotMatch(error.recovery, /自动重试中/);
	}
});
