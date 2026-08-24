/** P0-H 红测试（0824 总纲 §19.P0-H）：TUI 产品面。
 *
 * 红测试先行——三层信息密度/OSC8/命令冲突注册表不存在时必须红。
 *
 * 验收（文档原文）：
 * - 默认 transcript 中无 raw JSON（WP-7 已钉住）、无 [Extensions]
 *   （WP-7 快捷键注册表）、无 Pi update/engine 痕迹；
 * - 同一任务 Working… 占位新增次数为 0（单行就地更新）；
 * - 用户在 1 秒内能看见当前阶段（阶段化 working message——N9）；
 * - Worker/Operation 输出可在 Activity 展开（/logs 已钉住）；
 * - 命令名不与 Pi 内置冲突（/resume 事故的命令侧——快捷键注册表
 *   已覆盖键位，这里覆盖命令名）；
 * - artifact 可点击打开（OSC8 超链接）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

describe("P0-H 无 engine 痕迹", () => {
	it("Header/Footer 不含 Pi/engine 品牌字符串", async () => {
		const { renderHeader, renderFooter } = await import("../src/ui/product-state.js");
		const snap = {
			model: "Kimi K3", mode: "SIMULATION", body: "sim/ur5e",
			context_state: "Fresh", context_revision: 1,
			lease_state: "valid", operator: "READY",
			action_readiness: { state: "ready" },
			snapshot_seq: 1,
		} as never;
		for (const out of [renderHeader(snap, "zh-CN"), renderFooter(snap, "zh-CN")]) {
			assert.ok(!/pi-coding-agent|Pi SDK|powered by pi/i.test(out),
				`chrome 含 engine 痕迹: ${out.slice(0, 120)}`);
		}
	});

	it("命令注册表不与 Pi 内置命令冲突", async () => {
		const { ROSCLAW_COMMAND_NAMES, PI_BUILTIN_COMMANDS, commandConflicts } =
			await import("../src/extension/command-registry.js");
		assert.ok(ROSCLAW_COMMAND_NAMES.length > 0, "命令注册表为空");
		assert.deepEqual(commandConflicts(), [],
			`命令名冲突: ${commandConflicts()}`);
		assert.ok(!ROSCLAW_COMMAND_NAMES.includes("resume"),
			"/resume 与 Pi 内置冲突（WP-7 实证）——不得复活");
		assert.ok(PI_BUILTIN_COMMANDS.has("resume"), "Pi 内置表缺 resume");
	});
});

describe("P0-H 三层信息密度", () => {
	it("审计/治理类通知归 Debug 层，默认隐藏；切换后可见", async () => {
		const { NotificationLevelFilter } = await import("../src/ui/levels.js");
		const filter = new NotificationLevelFilter();
		assert.equal(filter.visible("debug"), false, "debug 层默认应隐藏");
		assert.equal(filter.visible("activity"), true);
		assert.equal(filter.visible("conversation"), true);
		filter.toggle();
		assert.equal(filter.visible("debug"), true, "切换后 debug 层不可见");
	});

	it("治理类文本分类为 debug（POLICY_AUTO/approval/grant 审计引用）", async () => {
		const { classifyNotice } = await import("../src/ui/levels.js");
		assert.equal(classifyNotice("安全仿真动作已自动放行执行（全程已记录审计）"), "activity");
		assert.equal(classifyNotice("approval apr_123 grant g_456 已消费"), "debug");
		assert.equal(classifyNotice("任务完成：验收 PASS"), "conversation");
	});
});

describe("P0-H artifact 可点击打开", () => {
	it("artifact 列表渲染 OSC8 超链接", async () => {
		const { mkdtempSync, writeFileSync } = await import("node:fs");
		const { tmpdir } = await import("node:os");
		const { join } = await import("node:path");
		const dir = mkdtempSync(join(tmpdir(), "p0h-"));
		const file = join(dir, "star.gif");
		writeFileSync(file, "GIF89a");
		const { renderArtifactList } = await import("../src/native/task-activity.js");
		const lines = renderArtifactList([{
			artifact_id: "art_1", path: file, media_type: "image/gif",
			sha256: "ab".repeat(32), size_bytes: 6,
		} as never]);
		const joined = lines.join("\n");
		assert.ok(joined.includes("]8;;"), "缺 OSC8 超链接序列——不可点击");
		assert.ok(joined.includes(encodeURIComponent("file") + "://" ) || joined.includes("file://"),
			"缺 file:// 链接目标");
	});
});

describe("P0-H Working 单行", () => {
	it("阶段化 working message 无换行（占位就地更新，新增=0）", async () => {
		const { phaseWorkingMessage } = await import("../src/extension/activity.js");
		for (const msg of [
			phaseWorkingMessage({ currentTool: "bash", operation: null }),
			phaseWorkingMessage({ currentTool: "", operation: { id: "op_1", label: "op_1" } }),
		]) {
			assert.ok(!msg.includes("\n"), `working message 多行: ${msg}`);
		}
	});
});
