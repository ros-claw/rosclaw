/** 0827 体验审计 P0-7 红测试：Provider quota 全局重试预算。
 *
 * 0827 实证：同一个 Kimi 403 配额错误——原始 JSON 显示多次 +
 * 规范化错误显示多次 + "Retry failed after 3 attempts"。
 *
 * 闭环断言：
 * 1. ProviderErrorGate：同一错误码只出一张中文错误卡（重试产生
 *    的重复 message_end 不重复打扰）；卡片含 /model 入口；
 * 2. "任务保持可恢复"只在有 active task 时出现；
 * 3. 原始错误进活动账本（onError 返回 activity 负载）；
 * 4. PROVIDER_PAUSED 状态：state-center snapshot.provider 翻转 +
 *    readiness reason_codes 带 PROVIDER_PAUSED；模型切换/下次
 *    成功清除（恢复同一 turn，不重建任务）。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("ProviderErrorGate：一码一卡 + /model 入口 + 条件可恢复", async () => {
	const { ProviderErrorGate } = await import("../src/native/model-errors.js");
	const { classifyModelError } = await import("../src/native/model-errors.js");
	const gate = new ProviderErrorGate();
	const quota = classifyModelError("403 quota exceeded: insufficient_quota");
	// 第一次：出卡 + activity 负载（raw 来自 message_end 的原始错误
	// 文本——与 extension 调用点同契约）。
	const first = gate.onError(quota, {
		hasActiveTask: true,
		raw: "403 quota exceeded: insufficient_quota",
	});
	assert.equal(first.showCard, true);
	assert.match(first.cardText, /PROVIDER_QUOTA_EXHAUSTED/);
	assert.match(first.cardText, /\/model/);
	assert.match(first.cardText, /任务保持可恢复/);
	assert.ok(first.activity, "缺 activity 负载（原始错误要进账本）");
	assert.match(String(first.activity?.raw), /quota/);
	// 同码重复（重试的 message_end）→ 不再出卡。
	const dup = gate.onError(quota, { hasActiveTask: true });
	assert.equal(dup.showCard, false, "同一错误码重复出卡");
	// 无 active task → 不说"任务保持可恢复"。
	const gate2 = new ProviderErrorGate();
	const noTask = gate2.onError(quota, { hasActiveTask: false });
	assert.equal(noTask.showCard, true);
	assert.doesNotMatch(noTask.cardText, /任务保持可恢复/);
	// 不同错误码 → 新卡。
	const other = gate.onError(
		classifyModelError("401 invalid api key"), { hasActiveTask: false },
	);
	assert.equal(other.showCard, true);
	// 模型切换/成功 → 闸门复位（恢复同一 turn）。
	gate.onModelSwitch();
	const afterSwitch = gate.onError(quota, { hasActiveTask: false });
	assert.equal(afterSwitch.showCard, true, "换模型后同码应重新出卡");
	gate.onModelSwitch();
	const gate3 = new ProviderErrorGate();
	gate3.onError(quota, { hasActiveTask: false });
	gate3.onSuccess();
	const afterSuccess = gate3.onError(quota, { hasActiveTask: false });
	assert.equal(afterSuccess.showCard, true, "成功后同码应重新出卡");
});

test("state-center：PROVIDER_PAUSED 进出快照与 readiness", async () => {
	const { ProductStateCenter } = await import("../src/session/state-center.js");
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
		contextLeaseId: "vcl_1",
		bodyId: "sim/ur5e",
		bodyHash: "body_x",
	});
	const center = new ProductStateCenter({
		rosclawHome: "/tmp/p07",
		active,
		operatorSocket: "/tmp/p07/run/operatord.sock",
		productVersion: "0.0.0-test",
		call: (async (_home: string, method: string) => {
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
		}) as never,
		operatorCallFn: async () => ({ ok: true }) as never,
	});
	await center.bootstrap();
	assert.equal(center.snapshot().provider ?? "OK", "OK");
	center.noteProviderPaused("PROVIDER_QUOTA_EXHAUSTED");
	assert.equal(center.snapshot().provider, "PAUSED");
	const readiness = center.snapshot().action_readiness;
	assert.ok(
		readiness.reason_codes.includes("PROVIDER_PAUSED"),
		`readiness 缺 PROVIDER_PAUSED：${readiness.reason_codes}`,
	);
	center.noteProviderOk();
	assert.equal(center.snapshot().provider, "OK");
});
