/** PR-H7 测试：provider 错误分类（总纲 v2 §8.4 + Gate F 子集）。 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { classifyModelError } from "../src/native/model-errors.js";

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
