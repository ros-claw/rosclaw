/** 十审 Gate W0 红测试（P0-WORKER-BLOCK / P0-ORDER-CORRELATION）：
 *
 * 1. delegate_returns_handle_without_awaiting_completion——工具不得
 *    同步等待整个任务；STARTED 响应必须带精确 WorkOrder ID（第一屏
 *    ID/worker/预算/deadline 由 agentd summary 承载，details 带结构化 ID）。
 * 2. delegate 不再做 mission 级轮询（pi.worker.status 取最后一单是
 *    错误的关联方式）。
 * 3. abort_propagates_to_cancel——signal abort 必须按精确 ID 发
 *    rosclaw_cancel_work（响应尚未返回也能 cancel）。
 */

import assert from "node:assert/strict";
import test from "node:test";

async function makeCtx(
	calls: Array<{ method: string; tool?: string; args?: Record<string, unknown> }>,
	executeImpl: (toolName: string, args: Record<string, unknown>) => Promise<unknown>,
) {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const active = new ActiveSessionContext({
		sessionId: "pi_test",
		missionId: "mis_1",
		contextRevision: 3,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "FRESH",
		leaseState: "ACTIVE",
		actionsAllowed: true,
	});
	const center = {
		call: async (method: string, params?: unknown) => {
			const p = (params ?? {}) as { request?: { tool_name?: string; arguments?: Record<string, unknown> } };
			const tool = p.request?.tool_name ?? "";
			const args = p.request?.arguments ?? {};
			calls.push({ method, tool, args });
			if (method === "pi.tools.execute") return await executeImpl(tool, args);
			return { ok: true };
		},
	};
	return { active, center };
}

test("delegate 返回 STARTED + 精确 WorkOrder ID，不做 mission 级轮询", async () => {
	const calls: Array<{ method: string; tool?: string; args?: Record<string, unknown> }> = [];
	const ctx = await makeCtx(calls, async () => ({
		ok: true,
		result: {
			ok: true,
			status: "STARTED",
			summary:
				"已启动后台 Worker（不阻塞本会话）。\nWorkOrder: wo_placeholder\nWorker: worker:native:basic\n预算: wall_time 300s",
		},
	}));
	const { buildDelegateTool } = await import("../src/tools/delegate.js");
	const tool = buildDelegateTool({ rosclawHome: "/tmp/rh", ...ctx } as never);
	const started = Date.now();
	const result = await tool.execute("t1", { goal: "长任务" }, undefined, undefined, {} as never);
	const elapsed = Date.now() - started;
	assert.ok(elapsed < 1000, `delegate 阻塞了 ${elapsed}ms`);
	const text = result.content.map((b) => ("text" in b ? b.text : "")).join("");
	assert.match(text, /WorkOrder: wo_/);
	const details = result.details as { status?: string; work_order_id?: string };
	assert.equal(details.status, "STARTED");
	assert.match(details.work_order_id ?? "", /^wo_[0-9a-f]{16}$/);
	// 不再经 pi.worker.status 做 mission 级轮询（错误关联根因）。
	assert.ok(
		!calls.some((c) => c.method === "pi.worker.status"),
		`仍做 mission 级轮询: ${JSON.stringify(calls)}`,
	);
	// 请求必须携带预生成的 work_order_id（abort 闭环的前提）。
	const delegateCall = calls.find((c) => c.tool === "rosclaw_delegate");
	assert.ok(delegateCall, "未调用 rosclaw_delegate");
	assert.equal(delegateCall.args?.work_order_id, details.work_order_id);
});

test("abort 按精确 ID 发 rosclaw_cancel_work（响应未返回也生效）", async () => {
	const calls: Array<{ method: string; tool?: string; args?: Record<string, unknown> }> = [];
	let releaseExecute: (() => void) | undefined;
	const gate = new Promise<void>((resolve) => {
		releaseExecute = resolve;
	});
	const ctx = await makeCtx(calls, async (tool) => {
		if (tool === "rosclaw_delegate") {
			await gate; // agentd 还在跑——abort 必须在此之前就能 cancel
			return { ok: true, result: { ok: true, status: "CANCELLED", summary: "已取消" } };
		}
		return { ok: true, result: { ok: true, status: "CANCELLED", summary: "已取消" } };
	});
	const { buildDelegateTool } = await import("../src/tools/delegate.js");
	const tool = buildDelegateTool({ rosclawHome: "/tmp/rh", ...ctx } as never);
	const controller = new AbortController();
	const pending = tool.execute("t1", { goal: "长任务" }, controller.signal, undefined, {} as never);
	// 等 delegate 请求发出。
	for (let i = 0; i < 100 && !calls.some((c) => c.tool === "rosclaw_delegate"); i++) {
		await new Promise((resolve) => setTimeout(resolve, 5));
	}
	const delegateCall = calls.find((c) => c.tool === "rosclaw_delegate");
	assert.ok(delegateCall, "delegate 请求未发出");
	const workOrderId = String(delegateCall.args?.work_order_id ?? "");
	assert.match(workOrderId, /^wo_[0-9a-f]{16}$/);
	controller.abort();
	// cancel 必须按同一个精确 ID 发出（不等 delegate 响应）。
	for (let i = 0; i < 100 && !calls.some((c) => c.tool === "rosclaw_cancel_work"); i++) {
		await new Promise((resolve) => setTimeout(resolve, 5));
	}
	const cancelCall = calls.find((c) => c.tool === "rosclaw_cancel_work");
	assert.ok(cancelCall, `abort 未触发 cancel: ${JSON.stringify(calls)}`);
	assert.equal(cancelCall.args?.work_order_id, workOrderId);
	releaseExecute?.();
	await pending;
});

test("check/cancel 工具按精确 ID 直发对应 agentd 工具", async () => {
	const calls: Array<{ method: string; tool?: string; args?: Record<string, unknown> }> = [];
	const ctx = await makeCtx(calls, async (tool) => ({
		ok: true,
		result: { ok: true, status: tool === "rosclaw_check_work" ? "RUNNING" : "CANCELLED", summary: "x" },
	}));
	const { buildCheckWorkTool, buildCancelWorkTool } = await import("../src/tools/delegate.js");
	const check = buildCheckWorkTool({ rosclawHome: "/tmp/rh", ...ctx } as never);
	await check.execute("t1", { work_order_id: "wo_abc12345" }, undefined, undefined, {} as never);
	const cancel = buildCancelWorkTool({ rosclawHome: "/tmp/rh", ...ctx } as never);
	await cancel.execute(
		"t2",
		{ work_order_id: "wo_abc12345", reason: "user" },
		undefined,
		undefined,
		{} as never,
	);
	assert.ok(calls.some((c) => c.tool === "rosclaw_check_work" && c.args?.work_order_id === "wo_abc12345"));
	assert.ok(calls.some((c) => c.tool === "rosclaw_cancel_work" && c.args?.work_order_id === "wo_abc12345"));
});
