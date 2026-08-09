/** PR-SIX-1 红测试（六审 §2/§3）：Single Control Plane + ProductSnapshotV2。
 *
 * 红测试先行——以下缺陷修复前必须红：
 *
 * 1. rosclaw_status 走 UDS pi.status（不再 HTTP 127.0.0.1:8765）；
 * 2. Header/Footer/status tool 共享同一 snapshot（同 seq、同 model/
 *    operator）——不允许顶部 Kimi K3/OFFLINE 底部未选模型/UNKNOWN；
 * 3. 显式 --mission 经 coordinator.attachInitialMission——leaseState
 *    写回 ACTIVE（Action 不再假 LOCKED）；
 * 4. readiness BLOCKED 时 request_action 零桥调用零副作用；
 * 5. lease lost / context stale 触发统一 chrome 刷新（subscribe）。
 */

import assert from "node:assert/strict";
import test from "node:test";

async function makeActive() {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	return new ActiveSessionContext({
		sessionId: "pi_test",
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
	});
}

// ------------------------------------------------------------ 红 1/2
test("rosclaw_status 经 UDS pi.status 读取共享快照（不是 HTTP 8765）", async () => {
	const { buildStatusTool } = await import("../src/tools/status.js");
	const calls: Array<{ method: string }> = [];
	const center = await makeCenter(calls);
	const tool = buildStatusTool(center as never);
	const result = await tool.execute("t1", {}, undefined, undefined, {} as never);
	const text = result.content.map((b) => ("text" in b ? b.text : "")).join("");
	// 必须经 UDS pi.status（注入的 call 只认 pi.status）。
	assert.ok(
		calls.some((c) => c.method === "pi.status"),
		`status 工具未走 pi.status UDS: ${JSON.stringify(calls)}`,
	);
	// 输出不得再出现旧 HTTP 面。
	assert.ok(!text.includes("127.0.0.1:8765"), `仍引用旧 HTTP 面: ${text}`);
	assert.ok(!text.includes("UNREACHABLE"), `UDS 可达却报 UNREACHABLE: ${text}`);
	assert.match(text, /READY/);
	// 结构化 details：与快照同 seq。
	const details = result.details as { snapshot_seq?: number; kernel?: string };
	assert.equal(details.kernel, "READY");
	assert.equal(typeof details.snapshot_seq, "number");
});

test("Header/Footer 共享同一 snapshot（同 seq 同 model 同 operator）", async () => {
	const calls: Array<{ method: string }> = [];
	const center = await makeCenter(calls);
	center.noteModel("Fake K3");
	await center.probeOperator();
	const { renderHeader, renderFooter } = await import("../src/ui/product-state.js");
	const snap = center.snapshot();
	const header = renderHeader(snap);
	const footer = renderFooter(snap);
	// 同一快照渲染——model/operator 两侧一致。
	assert.match(header, /Fake K3/);
	assert.match(footer, /Fake K3/, "Footer 不得落后/分叉于 Header 的 model");
	const operatorInHeader = /Operator (READY|OFFLINE|UNKNOWN)/.exec(header)?.[1];
	assert.ok(operatorInHeader);
	assert.ok(
		footer.includes(`Operator ${operatorInHeader}`),
		`Footer operator 与 Header 不一致: header=${operatorInHeader} footer=${footer}`,
	);
});

// ------------------------------------------------------------ 红 3
test("显式 mission 经 coordinator.attachInitialMission：leaseState 写回 ACTIVE", async () => {
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
	const active = await makeActive();
	active.patch({ leaseState: "NONE", contextState: "LOADING", actionsAllowed: false });
	const calls: Array<{ method: string; params: unknown }> = [];
	const call = async (_home: string, method: string, params: unknown) => {
		calls.push({ method, params });
		if (method === "pi.context") {
			// 合法 envelope（真实 hash/TTL——走真实校验路径）。
			const { envelopeHash } = await import("../src/extension/context-injection.js");
			const envelope = {
				schema_version: "rosclaw.embodied_context.v1",
				mission_id: "mis_1",
				context_revision: 4,
				generated_at: new Date().toISOString(),
				expires_at: new Date(Date.now() + 30_000).toISOString(),
				body: { body_id: "sim/ur5e", effective_body_hash: "body_x" },
				safety: { mode: "SIMULATION" },
				pending_approvals: [],
				hash: "",
			};
			envelope.hash = envelopeHash(envelope as never);
			return { ok: true, context: envelope, context_lease_id: "vcl_new" };
		}
		return { ok: true };
	};
	const leaseManager = new SessionLeaseManager("/tmp/rh-six1", call as never);
	// mock bind：getter 只能经 defineProperty 覆写（本测试只验证
	// coordinator 写回语义，不起真实 heartbeat）。
	leaseManager.bind = async () => {
		Object.defineProperty(leaseManager, "active", {
			get: () => ({ bindingId: "bnd_1" }),
			configurable: true,
		});
		return { bindingId: "bnd_1" } as never;
	};
	leaseManager.release = async () => undefined;
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: "/tmp/rh-six1",
		active,
		leaseManager,
		notify: () => undefined,
		call: call as never,
	});
	assert.equal(
		typeof (coordinator as { attachInitialMission?: unknown }).attachInitialMission,
		"function",
		"coordinator 缺 attachInitialMission——main.ts 只能绕过事务直接 bind",
	);
	const outcome = await coordinator.attachInitialMission("pi_test", "mis_1");
	assert.ok(outcome.ok, `attachInitialMission 失败: ${JSON.stringify(outcome)}`);
	const state = active.current;
	assert.equal(state.leaseState, "ACTIVE", "leaseState 未写回 ACTIVE——Action 假 LOCKED");
	assert.equal(state.contextState, "FRESH");
	assert.equal(state.actionsAllowed, true);
});

// ------------------------------------------------------------ 红 4
test("readiness BLOCKED 时 request_action 零桥调用返回 ACTION_LOCKED", async () => {
	const { buildRequestActionTool } = await import("../src/tools/request-action.js");
	const active = await makeActive();
	// lease 丢失 → 必须 BLOCKED。
	active.patch({ leaseState: "LOST", contextLeaseId: undefined });
	let bridgeCalled = false;
	const calls: Array<{ method: string }> = [];
	const center = await makeCenter(calls, active);
	// center.call 也被算入——readiness 探测用 operatorCallFn（不触碰桥）。
	const originalCall = center.call.bind(center);
	center.call = async (method: string, params?: Record<string, unknown>) => {
		if (String(method).startsWith("pi.action")) bridgeCalled = true;
		return originalCall(method, params);
	};
	const tool = buildRequestActionTool({
		rosclawHome: "/tmp/rh-six1",
		active,
		center: center as never,
	} as never);
	const result = await tool.execute(
		"t2",
		{ capability_id: "x.y", arguments: {} },
		undefined,
		undefined,
		{} as never,
	);
	assert.equal(bridgeCalled, false, "BLOCKED 时竟发起动作桥调用（会产生副作用）");
	const text = result.content.map((b) => ("text" in b ? b.text : "")).join("");
	assert.match(text, /ACTION_LOCKED/);
	const details = result.details as { status?: string; reason_codes?: string[] };
	assert.equal(details.status, "REJECTED");
	assert.ok(details.reason_codes?.includes("NO_WRITER_LEASE"));
});

// ------------------------------------------------------------ 红 5
test("ActiveSessionContext.subscribe：patch/replace/leaseLost 触发统一通知", async () => {
	const active = await makeActive();
	let fired = 0;
	(
		active as { subscribe?: (cb: () => void) => void }
	).subscribe?.(() => {
		fired += 1;
	});
	assert.ok(fired !== undefined, "subscribe 不存在");
	active.patch({ contextRevision: 5 });
	active.markLeaseLost();
	assert.ok(fired >= 2, `subscribe 未触发（fired=${fired}）——chrome 不会刷新`);
});

// ---------------------------------------------------------- 测试夹具
async function makeCenter(calls: Array<{ method: string }>, activeOverride?: unknown) {
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const active = activeOverride ?? (await makeActive());
	const call = async (_home: string, method: string, _params?: unknown) => {
		calls.push({ method });
		if (method === "pi.status") {
			return {
				ok: true,
				agentd: "READY",
				authorization_profile: "dev",
				mission: {
					mission_id: "mis_1",
					state: "ACTIVE",
					mode: "SIMULATION",
					body_id: "sim/ur5e",
				},
			};
		}
		if (method === "approvals.list") return { ok: true, approvals: [] };
		return { ok: true };
	};
	return new ProductStateCenter({
		rosclawHome: "/tmp/rh-six1",
		active: active as never,
		operatorSocket: "/tmp/rh-six1/run/operatord.sock",
		productVersion: "1.2.0",
		call: call as never,
		operatorCallFn: async () => ({ ok: true }),
	});
}
