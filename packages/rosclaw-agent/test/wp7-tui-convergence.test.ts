/** WP-7 红测试（0823 审计 §四.WP-7）：TUI 收敛。
 *
 * 红测试先行——模块/改名不存在时必须红。
 *
 * 0823 实测日志证据：
 * 1. 启动 [Extension issues]：/resume 命令与 ctrl+t 快捷键和 Pi
 *    内置冲突（快捷键被跳过——任务活动视图根本打不开）；
 * 2. 工具结果整段 JSON 原文刷屏（plan/trace 全字段）；
 * 3. propose_ur5e__execute_plan / "grant 已消费" 治理术语在 SIM
 *    用户界面可见；
 * 4. 会话被第一条消息 "hello" 命名——应命名自首个真实任务；
 * 5. REJECTED 与 "完成" 展示必须同源——REJECTED 的调用不得渲染
 *    出任何"完成"字样。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

describe("WP-7 中央快捷键注册表", () => {
	it("注册表存在且无冲突、无重复", async () => {
		const { ROSCLAW_SHORTCUTS, PI_RESERVED_SHORTCUTS, shortcutConflicts } =
			await import("../src/extension/shortcuts.js");
		const keys = Object.values(ROSCLAW_SHORTCUTS);
		assert.equal(new Set(keys).size, keys.length, "快捷键重复");
		assert.deepEqual(shortcutConflicts(), [], "与 Pi 内置保留键冲突");
		for (const key of keys) {
			assert.ok(!PI_RESERVED_SHORTCUTS.has(key), `${key} 是 Pi 内置保留键`);
		}
	});

	it("任务活动视图改用非冲突键（ctrl+t 是 Pi 内置）", async () => {
		const { ROSCLAW_SHORTCUTS } = await import("../src/extension/shortcuts.js");
		assert.notEqual(ROSCLAW_SHORTCUTS.taskActivity, "ctrl+t");
		assert.ok(ROSCLAW_SHORTCUTS.taskActivity.length > 0);
	});
});

describe("WP-7 命令不撞 Pi 内置", () => {
	it("/resume 改名（Pi 内置 /resume 冲突）", async () => {
		const { buildCommandHandlers } = await import("../src/extension/commands.js");
		const stub = {
			rosclawHome: "/tmp/wp7",
			active: { current: {}, subscribe: () => undefined },
			center: { call: async () => ({}), statusReport: async () => ({}) },
			locale: { effective: "zh" },
			registeredToolNames: () => [],
		};
		// @ts-expect-error stub 只覆盖本测试触及的面
		const handlers = buildCommandHandlers(stub);
		assert.ok(!("resume" in handlers), "/resume 与 Pi 内置冲突——必须改名");
		assert.ok("switch" in handlers, "改名后的会话切换命令缺失（/switch）");
	});
});

describe("WP-7 工具结果折叠与用户面显示", () => {
	it("SUCCEEDED envelope → 单行摘要，不刷原始 JSON", async () => {
		const { summarizeToolResultText } = await import("../src/ui/tool-display.js");
		const raw = JSON.stringify({
			status: "SUCCEEDED",
			capability_id: "ur5e.plan_cartesian_path",
			value: {
				ok: true,
				plan_id: "plan_1da59abd07f44831",
				summary: "star5：中心 (0.4, 0.0, 0.2)m，51 个插值点，已闭合",
				point_count: 51,
				points: Array.from({ length: 51 }, (_, i) => ({ x: i, y: i, z: 0.2 })),
			},
		});
		const out = summarizeToolResultText(raw);
		assert.ok(out.length < 200, `摘要过长（${out.length} 字符）——仍在刷屏`);
		assert.ok(out.includes("plan_1da59abd07f44831") || out.includes("已闭合"));
		assert.ok(!out.includes('"points"'), "原始 JSON 字段泄漏到用户面");
	});

	it("REJECTED → 明确'未执行'，绝不渲染'完成'", async () => {
		const { summarizeToolResultText } = await import("../src/ui/tool-display.js");
		const out = summarizeToolResultText(
			"REJECTED [DOOM_LOOP]: 同一调用已失败过一次——原样重复不会成功。",
		);
		assert.ok(out.includes("未执行"), "REJECTED 未标明未执行");
		assert.ok(out.includes("DOOM_LOOP"));
		assert.ok(!/已完成|执行完毕|COMPLETED/.test(out), "REJECTED 渲染出完成字样");
	});

	it("propose_/双下划线治理术语在用户面显示名中隐藏", async () => {
		const { displayLabelFor } = await import("../src/ui/tool-display.js");
		assert.equal(
			displayLabelFor("propose_ur5e__execute_plan"),
			"ur5e.execute_plan",
		);
		assert.equal(
			displayLabelFor("ur5e__get_cartesian_trace"),
			"ur5e.get_cartesian_trace",
		);
	});

	it("SIM auto 通知不含 POLICY_AUTO/approval/grant 术语", async () => {
		const { formatPolicyAutoNotice } = await import("../src/ui/tool-display.js");
		const out = formatPolicyAutoNotice({ approvalId: "apr_123" });
		assert.ok(!out.includes("POLICY_AUTO"), "治理术语泄漏");
		assert.ok(!out.includes("apr_123"), "approval id 泄漏到用户面");
		assert.ok(!/grant/i.test(out), "grant 术语泄漏");
		assert.ok(out.includes("仿真"), "缺少用户可理解的说明");
	});
});

describe("WP-7 会话命名自首个真实任务", () => {
	it("问候不命名；首个驱动工具活动的任务消息命名", async () => {
		const { AutoNamer } = await import("../src/session/auto-name.js");
		const namer = new AutoNamer();
		namer.noteInput("hello");
		assert.equal(namer.name(), null, "无工具活动的闲聊被用来命名");
		namer.noteInput("帮我用机械臂画一个五角星");
		assert.equal(namer.name(), null, "任务消息本身还未证明是真实任务");
		namer.noteToolActivity();
		assert.equal(namer.name(), "帮我用机械臂画一个五角星");
	});

	it("命名冻结：后续消息不改名；超长截断", async () => {
		const { AutoNamer } = await import("../src/session/auto-name.js");
		const namer = new AutoNamer();
		namer.noteInput("画五角星");
		namer.noteToolActivity();
		assert.equal(namer.name(), "画五角星");
		namer.noteInput("x".repeat(80));
		assert.equal(namer.name(), "画五角星", "命名后被后续消息覆盖");
		const namer2 = new AutoNamer();
		namer2.noteInput("y".repeat(80));
		namer2.noteToolActivity();
		assert.ok((namer2.name() ?? "").length <= 30, "命名未截断到 30 字");
	});
});

describe("WP-7 物化工具用户面 label", () => {
	it("propose_only 工具 label 不带 propose_ 前缀", async () => {
		const { materializeCapabilityTools } = await import("../src/tools/materialize.js");
		const snapshot = {
			schema_version: "rosclaw.capability_snapshot.v1",
			generation: 1,
			digest: "sha256:test",
			body_id: "sim/ur5e",
			mode: "SIMULATION",
			active: [{
				tool_name: "propose_ur5e__execute_plan",
				capability_id: "ur5e.execute_plan",
				exposure: "propose_only",
				effect_class: "SIMULATION_STATE_ONLY",
				description: "execute plan",
				input_schema: { type: "object" },
				output_schema: { type: "object" },
			}],
			excluded: [],
		};
		const ctx = {
			center: { call: async () => ({}) },
			active: { current: {} },
			rosclawHome: "/tmp/wp7",
		};
		// @ts-expect-error stub
		const tools = materializeCapabilityTools(snapshot, ctx);
		assert.equal(tools.length, 1);
		assert.equal(tools[0].name, "propose_ur5e__execute_plan", "工具名（模型调用面）不应改");
		assert.ok(!String(tools[0].label).includes("propose_"), "label 含治理前缀");
		assert.ok(!String(tools[0].label).includes("__"), "label 含双下划线");
	});
});
