/** PR-N9 红测试（调整方案 §八）：模型/会话/TUI 产品化。
 *
 * 红测试先行——以下不存在时必须红：
 * 1. /effort auto|low|medium|high 真实切换 reasoning effort；
 * 2. /sessions 与 /switch 命令存在（TUI 会话面；WP-7：/resume 与
 *    Pi 内置冲突改名 /switch）；
 * 3. 纯 SIM 下不展示 OPERATOR_OFFLINE（SIM 不需要 operator）；
 * 4. Working… 被结构化阶段替代（工具/操作进行中显示当前阶段，
 *    不是静态 Working…）——展示可审计事件，不展示思维链。
 */
import assert from "node:assert/strict";
import { test } from "node:test";

test("N9: /effort 与 /sessions、/switch 命令存在", async () => {
	const { buildCommandHandlers } = await import("../src/extension/commands.js");
	const handlers = buildCommandHandlers({
		rosclawHome: "/tmp/x",
		active: undefined as never,
		center: undefined as never,
		locale: undefined as never,
		registeredToolNames: () => [],
	});
	assert.ok(handlers.effort, "/effort 缺失");
	assert.ok(handlers.sessions, "/sessions 缺失");
	assert.ok(handlers.switch, "/switch 缺失");
});

test("N9: /effort 接受 auto|low|medium|high 并真实应用", async () => {
	const { buildCommandHandlers } = await import("../src/extension/commands.js");
	const applied: string[] = [];
	let notified = "";
	const handlers = buildCommandHandlers({
		rosclawHome: "/tmp/x",
		active: undefined as never,
		center: undefined as never,
		locale: undefined as never,
		registeredToolNames: () => [],
	});
	const ctx = {
		ui: { notify: (msg: string) => { notified = msg; } },
		setThinkingLevel: (level: string) => { applied.push(level); },
	} as never;
	await handlers.effort.handler("high", ctx);
	assert.deepEqual(applied, ["high"]);
	assert.match(notified, /high/i);
	await handlers.effort.handler("bogus", ctx);
	assert.match(notified, /auto\|low\|medium\|high/);
});

test("N9: 纯 SIM 不展示 Operator Offline（renderOperator 唯一 seam）", async () => {
	const { renderOperator } = await import("../src/ui/product-state.js");
	// 纯 SIM：operator 状态降为不展示（PR-14.7 语义钉住）。
	const sim = renderOperator({
		mode: "SIMULATION", operator: "OFFLINE",
		action_readiness: { state: "READY", reason_codes: [] },
	} as never, "zh-CN" as never);
	assert.ok(!/离线|Offline|OFFLINE/.test(sim), `纯 SIM 仍显示 Offline: ${sim}`);
	// REAL 下必须显示。
	const real = renderOperator({
		mode: "REAL", operator: "OFFLINE",
		action_readiness: { state: "BLOCKED", reason_codes: ["OPERATOR_OFFLINE"] },
	} as never, "zh-CN" as never);
	assert.ok(/离线|Offline|OFFLINE/.test(real), `REAL 下 Offline 未显示: ${real}`);
});

test("N9: Working… 被结构化阶段替代", async () => {
	const { phaseWorkingMessage } = await import("../src/extension/activity.js");
	// 有当前工具/操作 → 显示阶段，不是静态 Working…。
	const phase = phaseWorkingMessage({
		currentTool: "ur5e__simulate_cartesian_trajectory",
		operation: null,
	});
	assert.match(phase, /ur5e__simulate/);
	assert.ok(!/^Working…$/.test(phase));
	const opPhase = phaseWorkingMessage({
		currentTool: null,
		operation: { id: "op_1", label: "MuJoCo 动力学仿真", detail: "730/7307 steps" },
	});
	assert.match(opPhase, /动力学仿真/);
	assert.match(opPhase, /730\/7307/);
	// 无活动时回到默认。
	assert.equal(phaseWorkingMessage({ currentTool: null, operation: null }), "Working…");
});
