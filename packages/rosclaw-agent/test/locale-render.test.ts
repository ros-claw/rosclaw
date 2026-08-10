/** PR-SEVEN-6 红测试（七审 §6 i18n/TUI 部分）：
 *
 * 1. 普通层不显示 `OPERATOR_OFFLINE（operator offline）` 双重双语——
 *    Header 只显示本地化原因（code 在 /status/JSON）；
 * 2. ROSCLAW_TUI_UNICODE=never 或 LC_ALL=C 时，分隔符用 ASCII（不
 *    渲染 ·/—/→ 为 ?）。
 */

import assert from "node:assert/strict";
import test from "node:test";

function makeSnap(operator: string) {
	return {
		snapshot_seq: 3,
		product_version: "1.2.0",
		kernel: "READY",
		model: "Fake K3",
		mode: "SIMULATION",
		mission_id: "mis_abc",
		body_id: "sim/ur5e",
		context_state: "FRESH",
		context_revision: 0,
		lease_state: "ACTIVE",
		operator,
		action_readiness: {
			state: "BLOCKED",
			reason_codes: ["OPERATOR_OFFLINE"],
			snapshot_seq: 3,
		},
	} as never;
}

test("受阻原因不双重双语（无 CODE（label） 形式）", async () => {
	const { renderHeader } = await import("../src/ui/product-state.js");
	const header = renderHeader(makeSnap("OFFLINE"), "zh-CN");
	assert.ok(!/OPERATOR_OFFLINE（/.test(header), `双重双语: ${header}`);
	assert.ok(!/OPERATOR_OFFLINE\(operator offline\)/.test(header));
	// 中文层显示本地化原因。
	assert.match(header, /操作员离线|操作员 离线/);
});

test("非 UTF-8 终端用 ASCII 分隔符", async () => {
	const { renderHeader, renderFooter } = await import("../src/ui/product-state.js");
	process.env.ROSCLAW_TUI_UNICODE = "never";
	try {
		const header = renderHeader(makeSnap("OFFLINE"), "en-US");
		const footer = renderFooter(makeSnap("OFFLINE"), "en-US");
		assert.ok(!header.includes("·"), `header 含中点: ${header}`);
		assert.ok(!footer.includes("·"), `footer 含中点: ${footer}`);
		assert.ok(!/[—→]/.test(header + footer), "含非 ASCII 图形字符");
		assert.ok(header.includes("|") || header.includes(" - "), "ASCII 分隔符缺失");
	} finally {
		delete process.env.ROSCLAW_TUI_UNICODE;
	}
});

test("默认（UTF-8）仍用 · 分隔", async () => {
	const { renderHeader } = await import("../src/ui/product-state.js");
	delete process.env.ROSCLAW_TUI_UNICODE;
	const header = renderHeader(makeSnap("OFFLINE"), "en-US");
	assert.ok(header.includes("·"), "UTF-8 默认应用中点分隔");
});
