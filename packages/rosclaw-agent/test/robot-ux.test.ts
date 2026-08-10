/** PR-SEVEN-5 红测试（七审 §6 Robot-first UX）：
 *
 * 1. Header 显示机器人友好名（body_display），不再是裸内部 body_id；
 * 2. zh-CN 界面用"机器人"而不是内部概念 Body/本体；
 * 3. /robot /robots /capabilities /doctor task 等新命令已注册
 *    （InputGuard 放行 + handlers 存在）。
 */

import assert from "node:assert/strict";
import test from "node:test";

function makeSnap() {
	return {
		snapshot_seq: 3,
		product_version: "1.2.0",
		kernel: "READY",
		model: "Fake K3",
		mode: "SIMULATION",
		mission_id: "mis_abc",
		body_id: "sim/ur5e",
		body_display: "UR5e（本地仿真）",
		context_state: "FRESH",
		context_revision: 0,
		lease_state: "ACTIVE",
		operator: "READY",
		action_readiness: { state: "READY", reason_codes: [], snapshot_seq: 3 },
	} as never;
}

test("Header 显示机器人友好名", async () => {
	const { renderHeader } = await import("../src/ui/product-state.js");
	const header = renderHeader(makeSnap(), "zh-CN");
	assert.ok(header.includes("UR5e（本地仿真）"), `缺友好名: ${header}`);
	assert.ok(!header.includes("sim/ur5e"), `仍显示内部 body_id: ${header}`);
});

test("zh-CN 用『机器人』标签", async () => {
	const { renderHeader } = await import("../src/ui/product-state.js");
	const header = renderHeader(makeSnap(), "zh-CN");
	assert.match(header, /机器人 UR5e/);
	const en = renderHeader(makeSnap(), "en-US");
	assert.match(en, /Robot UR5e/);
});

test("Robot-first 命令已注册并放行", async () => {
	const { buildCommandHandlers } = await import("../src/extension/commands.js");
	const { guardInput } = await import("../src/extension/input-guard.js");
	const deps = {
		center: { call: async () => ({ ok: true }), probeOperator: async () => "READY" },
		active: { current: {} },
		locale: { effective: "zh-CN", current: { ui_locale: "auto", reply_language: "follow-user" } },
	} as never;
	const handlers = buildCommandHandlers(deps);
	for (const name of ["robot", "robots", "capabilities", "doctor"]) {
		assert.ok(handlers[name], `缺 /${name} handler`);
		const guard = guardInput(`/${name}`, "developer");
		assert.equal(guard.action, "continue", `/${name} 被 InputGuard 拦截`);
	}
});
