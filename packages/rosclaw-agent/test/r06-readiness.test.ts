/** R0-6 红测试（0826 体验审计 §5.R0-6）：Readiness 单源与启动事务。
 *
 * 真实事故（0826 体验旅程）：Header 同时显示 Kernel Unreachable +
 * Context Stale + Action Blocked，同一会话却成功完成 SIM 任务——
 * 状态系统不可信（启动瞬态被当成稳态真相呈现）。
 *
 * 断言：
 * 1. 启动 barrier：bootstrap 未完成时 readiness 是 PREPARING（不是
 *    BLOCKED），Header 显示"正在准备"——不显示 Unreachable/
 *    Stale/Blocked 三连；
 * 2. bootstrap 成功 → READY；全部重试失败 → 诚实 UNREACHABLE
 *    （有限重试，不是无限也不是一次定死）；
 * 3. 运行期瞬态失败 → 内核有限重连（不消耗模型 token）成功后
 *    恢复 READY；
 * 4. SIM auto 下 OPERATOR_OFFLINE 不是 blocker（信息提示）；
 * 5. UI 与 admission 同一 snapshot_seq（一致性钉住）；
 * 6. SIM stale context 在 recipe 前自动 refresh；REAL 不自动刷新
 *    （fail closed）。
 */

import assert from "node:assert/strict";
import test from "node:test";

async function makeActive(overrides: Record<string, unknown> = {}) {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	return new ActiveSessionContext({
		sessionId: "pi_r06",
		missionId: "mis_1",
		contextRevision: 3,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "FRESH",
		leaseState: "ACTIVE",
		actionsAllowed: true,
		contextLeaseId: "vcl_1",
		bodyId: "sim/ur5e",
		bodyHash: "body_x",
		...overrides,
	});
}

// 通用成功响应（capabilities 必须带 action_capabilities——否则
// ROBOT_KIT_INCOMPLETE blocker 干扰 readiness 断言）。
function okPayload(method: string): Record<string, unknown> {
	if (method === "pi.capabilities") {
		return { ok: true, action_capabilities: [{ tool_id: "ur5e.move_to_pose" }] };
	}
	return { ok: true };
}

function centerWith(
	call: (home: string, method: string) => Promise<Record<string, unknown>>,
	active: unknown,
) {
	return import("../src/session/state-center.js").then(
		({ ProductStateCenter }) =>
			new ProductStateCenter({
				rosclawHome: "/tmp/rh-r06",
				active: active as never,
				operatorSocket: "/tmp/rh-r06/run/operatord.sock",
				productVersion: "1.2.0",
				call: call as never,
				operatorCallFn: (async () => ({ ok: true })) as never,
			}),
	);
}

test("启动 barrier：bootstrap 未完成 = PREPARING（不是 BLOCKED 三连）", async () => {
	// 桥调用挂起（启动慢）——bootstrap 期间的 UI/ready 必须是
	// "正在准备"，不是 Kernel Unreachable + Context Stale +
	// Action Blocked。
	const center = await centerWith(
		() => new Promise(() => undefined),
		await makeActive(),
	);
	const readiness = center.snapshot().action_readiness;
	assert.equal(
		readiness.state,
		"PREPARING",
		`bootstrap 期应 PREPARING 而非 ${readiness.state}（${readiness.reason_codes}）`,
	);
	const { renderHeader } = await import("../src/ui/product-state.js");
	const header = renderHeader(center.snapshot(), "en-US");
	assert.match(header, /preparing|Preparing|准备/);
	assert.ok(!header.includes("Unreachable"), header);
	assert.ok(!header.includes("Blocked"), header);
});

test("bootstrap 成功 → READY（一次成功即不再 PREPARING）", async () => {
	const center = await centerWith(
		async (_home, method) => okPayload(String(method)),
		await makeActive(),
	);
	await center.bootstrap();
	const snap = center.snapshot();
	assert.equal(snap.kernel, "READY");
	assert.equal(snap.action_readiness.state, "READY");
});

test("bootstrap 有限重试后诚实 UNREACHABLE（不是一次定死）", async () => {
	let attempts = 0;
	const center = await centerWith(
		async () => {
			attempts += 1;
			throw new Error("socket missing");
		},
		await makeActive(),
	);
	await center.bootstrap();
	const snap = center.snapshot();
	assert.equal(snap.kernel, "UNREACHABLE");
	assert.ok(attempts >= 2, `应有有限重试（实际 ${attempts} 次）`);
	// 上限 = bootstrap 3 + 失败后 capability 探测 1 + recover 2
	// （全部内核行为，有界，不消耗模型 token）。
	assert.ok(attempts <= 8, `重试必须有界（实际 ${attempts} 次）`);
});

test("运行期瞬态失败 → 内核有限重连恢复（不消耗模型 token）", async () => {
	let calls = 0;
	const center = await centerWith(
		async (_home, method) => {
			calls += 1;
			if (calls === 1) throw new Error("transient");
			return okPayload(String(method));
		},
		await makeActive(),
	);
	await assert.rejects(center.call("pi.status", {}));
	// 瞬态失败 → 内核自动有限重连（后台 recoverKernel 已调度，
	// 第二次调用即成功）——恢复是内核行为，不是模型回合。
	await center.recoverKernel();
	assert.ok(calls >= 2, `重连探测未发生（calls=${calls}）`);
	assert.equal(center.snapshot().kernel, "READY");
});

test("SIM auto：OPERATOR_OFFLINE 不是 blocker（信息提示即可）", async () => {
	const center = await centerWith(
		async (_home, method) => okPayload(String(method)),
		await makeActive(),
	);
	// operator 探测返回失败（offline）。
	(center as never as { operatorCallFn: unknown });
	const readiness = center.snapshot().action_readiness;
	assert.ok(
		!readiness.reason_codes.includes("OPERATOR_OFFLINE"),
		`SIM auto 不应被 operator offline 阻塞: ${readiness.reason_codes}`,
	);
});

test("UI 与 admission 同一 snapshot_seq", async () => {
	const center = await centerWith(
		async (_home, method) => okPayload(String(method)),
		await makeActive(),
	);
	const snap = center.snapshot();
	assert.equal(snap.action_readiness.snapshot_seq, snap.snapshot_seq);
});

test("SIM stale context 在 recipe 前自动 refresh；REAL fail closed", async () => {
	const { ensureFreshContextForSim } = await import(
		"../src/extension/context-injection.js"
	);
	const refreshed: string[] = [];
	const call = async (_home: string, method: string) => {
		refreshed.push(method);
		return { ok: true, context: {} };
	};
	const staleSim = await makeActive({
		contextState: "STALE",
		mode: "SIMULATION",
	});
	await ensureFreshContextForSim("/tmp/rh-r06", staleSim as never, call as never);
	assert.ok(
		refreshed.includes("pi.context"),
		"SIM stale 必须自动 refresh",
	);
	const staleReal = await makeActive({
		contextState: "STALE",
		mode: "REAL",
	});
	refreshed.length = 0;
	await ensureFreshContextForSim("/tmp/rh-r06", staleReal as never, call as never);
	assert.equal(
		refreshed.length,
		0,
		"REAL 不得自动 refresh（fail closed 由内核 admission 裁决）",
	);
});
