/** P1-A1（0824 总纲 §10.1）：Pi probe 报告契约。
 *
 * - 未配置 → MODEL_NOT_CONFIGURED（指向 setup model），不发网络请求；
 * - auth 缺失 → AUTH_NOT_CONFIGURED；
 * - 四步全过 → reachable/chat_ok/tool_call_ok 全 true；
 * - 错误分类稳定（401→AUTH_FAILED、429→RATE_LIMITED）；
 * - 报告绝不包含 secret 材料。
 */

import assert from "node:assert/strict";
import test from "node:test";

import { probePiModel } from "../src/harness/pi/pi-probe.js";

interface FakeModel {
	provider: string;
	id: string;
}

function fakeRuntime(overrides: {
	auth?: boolean;
	available?: FakeModel[];
	modelsError?: Error;
	chatReply?: unknown;
	toolReply?: unknown;
	chatError?: Error;
	toolError?: Error;
}) {
	const model: FakeModel = { provider: "kimi-code", id: "k3" };
	return {
		hasConfiguredAuth: () => overrides.auth ?? true,
		getAvailable: async () => {
			if (overrides.modelsError) throw overrides.modelsError;
			return overrides.available ?? [model];
		},
		getModel: () => model,
		completeSimple: async (_m: unknown, context: { tools?: unknown[] }) => {
			if (context.tools && context.tools.length > 0) {
				if (overrides.toolError) throw overrides.toolError;
				return (
					overrides.toolReply ?? {
						stopReason: "toolUse",
						content: [{ type: "toolCall", name: "report_ok", arguments: { ok: true } }],
					}
				);
			}
			if (overrides.chatError) throw overrides.chatError;
			return (
				overrides.chatReply ?? {
					stopReason: "stop",
					content: [{ type: "text", text: "OK" }],
				}
			);
		},
	} as never;
}

const BASE = { agentDir: "/tmp/x", cwd: "/tmp/x", profile: "developer" as const };
const DEFAULTS = { provider: "kimi-code", model: "k3" };

test("未配置 defaultProvider/defaultModel → MODEL_NOT_CONFIGURED", async () => {
	const report = await probePiModel({ ...BASE, defaults: {} });
	assert.equal(report.reachable, false);
	assert.match(report.error ?? "", /MODEL_NOT_CONFIGURED/);
	assert.match(report.error ?? "", /setup model/);
});

test("auth 缺失 → AUTH_NOT_CONFIGURED，不做 chat", async () => {
	const report = await probePiModel({
		...BASE,
		defaults: DEFAULTS,
		runtime: fakeRuntime({ auth: false }),
	});
	assert.equal(report.auth_configured, false);
	assert.match(report.error ?? "", /AUTH_NOT_CONFIGURED/);
	assert.equal(report.chat_ok, false);
});

test("四步全过 → 全绿 + provider/model 回显", async () => {
	const report = await probePiModel({
		...BASE,
		defaults: DEFAULTS,
		runtime: fakeRuntime({}),
	});
	assert.equal(report.engine, "pi");
	assert.equal(report.reachable, true);
	assert.equal(report.chat_ok, true);
	assert.equal(report.tool_call_ok, true);
	assert.equal(report.expected_model_present, true);
	assert.equal(report.provider, "kimi-code");
	assert.equal(report.model, "k3");
	assert.deepEqual(report.models_visible, ["k3"]);
	assert.equal(report.error, undefined);
});

test("models listing 401 → AUTH_FAILED 分类", async () => {
	const report = await probePiModel({
		...BASE,
		defaults: DEFAULTS,
		runtime: fakeRuntime({ modelsError: new Error("HTTP 401 unauthorized") }),
	});
	assert.equal(report.reachable, false);
	assert.match(report.error ?? "", /^AUTH_FAILED/);
});

test("chat 429 → RATE_LIMITED 分类", async () => {
	const report = await probePiModel({
		...BASE,
		defaults: DEFAULTS,
		runtime: fakeRuntime({ chatError: new Error("429 rate limit exceeded") }),
	});
	assert.equal(report.reachable, true);
	assert.equal(report.chat_ok, false);
	assert.match(report.error ?? "", /^RATE_LIMITED/);
});

test("tool call 缺失 → TOOL_CALL_PROBE_FAILED（诚实失败）", async () => {
	const report = await probePiModel({
		...BASE,
		defaults: DEFAULTS,
		runtime: fakeRuntime({
			toolReply: { stopReason: "stop", content: [{ type: "text", text: "ok" }] },
		}),
	});
	assert.equal(report.chat_ok, true);
	assert.equal(report.tool_call_ok, false);
	assert.match(report.error ?? "", /TOOL_CALL_PROBE_FAILED/);
});

test("报告序列化不含 secret 材料", async () => {
	const report = await probePiModel({
		...BASE,
		defaults: DEFAULTS,
		runtime: fakeRuntime({}),
	});
	const text = JSON.stringify(report);
	assert.ok(!text.includes("sk-"), `报告含 key 材料: ${text}`);
});
