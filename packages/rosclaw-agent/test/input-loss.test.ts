/** NINE-1 红测试（九审 §0.1/§1.4）：P0-INPUT-LOSS。
 *
 * 红测试先行——九审实测：自然语言"我想用机械臂画五角星"在 input
 * hook 里被路由直跑并返回 handled，Pi prompt() 立即返回——输入
 * 不进模型、不进消息链、不落 session JSONL（幽灵执行）。
 *
 * 原则：先显示、先落账、先分配 Turn ID，然后才能路由/执行。
 * natural language 绝不许在 input hook 里 handled。
 */

import assert from "node:assert/strict";
import test from "node:test";

async function callInputHandler(text: string) {
	const { createRosclawExtension } = await import("../src/extension/index.js");
	const { envelopeHash } = await import("../src/extension/context-injection.js");
	const handlers: Record<string, (event: unknown, ctx: unknown) => Promise<unknown>> = {};
	const fakePi = new Proxy(
		{
			on(name: string, handler: never) {
				handlers[name] = handler;
			},
			events: { emit: async () => ({}) },
		},
		{
			get(target, prop) {
				if (prop in target) return (target as Record<string | symbol, unknown>)[prop];
				return () => undefined; // 其余 pi API 一律吞掉（测试只要 input handler）
			},
		},
	);
	const center = {
		call: async (method: string, _params: unknown) => {
			// 路由命中 + 任务成功——当前代码会据此返回 handled（红）；
			// 修复后即使全链可用也必须 continue。
			if (method === "pi.intent.route") {
				return { ok: true, spec: { goal: "draw_shape", parameters: {} } };
			}
			if (method === "pi.tools.execute") {
				return {
					ok: true,
					result: { summary: JSON.stringify({ state: "VERIFIED", user_view: "ok" }) },
				};
			}
			if (method === "pi.context") {
				const envelope = {
					schema_version: "rosclaw.embodied_context.v1",
					expires_at: new Date(Date.now() + 60_000).toISOString(),
				} as Record<string, unknown>;
				(envelope as { hash?: string }).hash = envelopeHash(envelope as never);
				return {
					ok: true,
					context: envelope,
					context_lease_id: "ctxl_mock",
					context_lease_expires_at: new Date(Date.now() + 60_000).toISOString(),
				};
			}
			return { ok: true };
		},
		probeOperator: async () => "OFFLINE",
		refreshCapabilities: async () => {},
		refreshRobotInfo: async () => {},
		snapshot: () => ({ robot_kit: { state: "READY" } }),
		isSimAutoPolicy: true,
		subscribe: () => () => {},
	};
	const options = {
		profile: "developer",
		version: "1.2.0",
		systemPrompt: "",
		active: {
			current: {
				missionId: "mis_1",
				sessionId: "pi_1",
				mode: "SIMULATION",
				contextLeaseId: "ctxl_1",
				contextRevision: 0,
			},
			subscribe: () => () => {},
			applyEnvelope() {},
		},
		coordinator: {},
		center,
		locale: { effective: "zh-CN", subscribe: () => () => {} },
		rosclawHome: "/tmp/rh-nine1",
	} as never;
	const factory = createRosclawExtension(options);
	factory(fakePi as never);
	const input = handlers["input"];
	assert.ok(input, "input handler 未注册");
	const ctx = { hasUI: true, ui: { notify() {}, setTitle() {}, setWidget() {}, setWorkingIndicator() {}, setWorkingMessage() {}, setHiddenThinkingLabel() {}, setHeader() {}, setFooter() {} } };
	return await input({ text }, ctx);
}

test("自然语言任务输入绝不返回 handled", async () => {
	for (const text of ["我想用机械臂画五角星", "画一个五角星", "帮我诊断 RealSense 断流"]) {
		const result = (await callInputHandler(text)) as { action: string };
		assert.notEqual(
			result.action,
			"handled",
			`输入被 handled 吞掉（P0-INPUT-LOSS）: ${text}`,
		);
		assert.equal(result.action, "continue");
	}
});
