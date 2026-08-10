/** PR-SIX-5 红测试（六审 §8）：i18n catalog + /language。
 *
 * 红测试先行：
 * 1. zh-CN 与 en-US catalog 键位必须一一对应（缺翻译 CI fail）；
 * 2. renderHeader/renderFooter 按 locale 渲染（zh=任务/机器人/操作员/
 *    动作；en=Mission/Robot/Operator/Action（PR-SEVEN-5：Body→机器人/Robot））——当前是散落硬编码；
 * 3. /language 命令存在且持久化（查看/中文/English/auto/lock）。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("catalog 键位 zh-CN/en-US 完全对齐", async () => {
	const { CATALOG_ZH } = await import("../src/i18n/catalog.zh-CN.js");
	const { CATALOG_EN } = await import("../src/i18n/catalog.en-US.js");
	const zhKeys = Object.keys(CATALOG_ZH).sort();
	const enKeys = Object.keys(CATALOG_EN).sort();
	assert.deepEqual(zhKeys, enKeys, `catalog 键位不一致: zh-only=${zhKeys.filter((k) => !enKeys.includes(k))} en-only=${enKeys.filter((k) => !zhKeys.includes(k))}`);
	for (const key of zhKeys) {
		assert.ok(String((CATALOG_ZH as Record<string, string>)[key]).trim(), `zh 空翻译: ${key}`);
		assert.ok(String((CATALOG_EN as Record<string, string>)[key]).trim(), `en 空翻译: ${key}`);
	}
});

test("renderHeader/renderFooter 按 locale 渲染同一快照", async () => {
	const { renderHeader, renderFooter } = await import("../src/ui/product-state.js");
	const snap = {
		snapshot_seq: 7,
		product_version: "1.2.0",
		kernel: "READY",
		model: "Fake K3",
		mode: "SIMULATION",
		mission_id: "mis_abc123",
		body_id: "sim/ur5e",
		context_state: "FRESH",
		context_revision: 3,
		lease_state: "ACTIVE",
		operator: "OFFLINE",
		action_readiness: {
			state: "BLOCKED",
			reason_codes: ["OPERATOR_OFFLINE"],
			snapshot_seq: 7,
		},
	} as never;
	const zhHeader = renderHeader(snap, "zh-CN");
	const enHeader = renderHeader(snap, "en-US");
	// 中文 chrome
	assert.match(zhHeader, /任务 mis_abc123/);
	assert.match(zhHeader, /机器人 sim\/ur5e/);  // PR-SEVEN-5：本体→机器人
	assert.match(zhHeader, /操作员 离线/);
	assert.match(zhHeader, /动作 受阻/);
	assert.match(zhHeader, /仿真/);
	// 英文 chrome
	assert.match(enHeader, /Mission mis_abc123/);
	assert.match(enHeader, /Robot sim\/ur5e/);  // PR-SEVEN-5：Body→Robot
	assert.match(enHeader, /Operator Offline/);
	assert.match(enHeader, /Action Blocked/i);
	assert.match(enHeader, /Simulation/);
	const zhFooter = renderFooter(snap, "zh-CN");
	const enFooter = renderFooter(snap, "en-US");
	assert.match(zhFooter, /仿真/);
	assert.match(enFooter, /Simulation/);
	assert.match(zhFooter, /Fake K3/);
	assert.match(enFooter, /Fake K3/);
});

test("/language 命令注册且持久化 locale", async () => {
	const { buildCommandHandlers } = await import("../src/extension/commands.js");
	const writes: Array<[string, string]> = [];
	const persisted: Record<string, string> = {};
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const active = new ActiveSessionContext({
		sessionId: "pi_test",
		missionId: undefined,
		contextRevision: 0,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "LOADING",
		leaseState: "NONE",
		actionsAllowed: false,
	});
	const center = new ProductStateCenter({
		rosclawHome: "/tmp/rh-six5",
		active,
		operatorSocket: "/tmp/rh-six5/run/operatord.sock",
		productVersion: "1.2.0",
		call: async () => ({ ok: true }),
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = {
		get current() {
			return {
				ui_locale: (persisted.ui_locale ?? "auto") as "auto",
				reply_language: "follow-user" as const,
			};
		},
		get effective(): "zh-CN" | "en-US" {
			const v = persisted.ui_locale ?? "auto";
			return v === "auto" ? "zh-CN" : (v as "zh-CN");
		},
		setUiLocale: (value: string) => {
			persisted.ui_locale = value;
			writes.push(["ui.locale", value]);
		},
		setReplyLanguage: (value: string) => {
			persisted.reply_language = value;
		},
	};
	const commands = buildCommandHandlers({
		rosclawHome: "/tmp/rh-six5",
		active,
		center,
		locale: locale as never,
		registeredToolNames: () => [],
	} as never);
	const language = commands.language;
	assert.ok(language, "/language 命令未注册");
	const notifications: string[] = [];
	const ctx = { ui: { notify: (m: string) => notifications.push(m) } };
	await language.handler("", ctx as never);
	assert.ok(notifications.at(-1)?.includes("auto"), "无参应显示当前策略");
	await language.handler("English", ctx as never);
	assert.equal(persisted.ui_locale, "en-US", "English 未持久化为 en-US");
	await language.handler("中文", ctx as never);
	assert.equal(persisted.ui_locale, "zh-CN", "中文未持久化为 zh-CN");
});
