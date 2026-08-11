/** PR-EIGHT-7 红测试（八审 §4 P0-9/P1.7）：状态投影一致性。
 *
 * 红测试先行——审计实测："Robot kit incomplete:  — One-key repair"
 * （reason 为空仍渲染悬空冒号）；kit READY 后旧 BROKEN warning 无
 * 正面清除。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("kit BROKEN 提示：空 reason 不渲染悬空冒号", async () => {
	const { formatKitBrokenHint } = await import("../src/ui/product-state.js");
	const hint = formatKitBrokenHint(
		{ state: "BROKEN", reason: "", remediation: { command: "/robot repair rosclaw/ur5e-sim" } },
		"zh-CN",
	);
	assert.ok(!/:\s*[—-]/.test(hint), `悬空冒号: ${hint}`);
	assert.ok(hint.includes("/robot repair"), hint);
	const withReason = formatKitBrokenHint(
		{ state: "BROKEN", reason: "executor MISSING", remediation: null },
		"zh-CN",
	);
	assert.ok(withReason.includes("executor MISSING"), withReason);
});

test("kit 恢复 READY 有正面清除文案", async () => {
	const { formatKitRecoveredHint } = await import("../src/ui/product-state.js");
	const hint = formatKitRecoveredHint("UR5e（本地仿真）", "zh-CN");
	assert.ok(hint.includes("UR5e"), hint);
	assert.match(hint, /就绪|可用|ready/i);
});
