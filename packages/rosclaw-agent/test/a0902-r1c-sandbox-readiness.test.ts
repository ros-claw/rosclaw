/** 0902 R1-c 红测试（§5.3）：无 OS 沙箱时，会话开始一次性提示——
 *  任务开始前让用户知道 shell 将走确认卡降级 + doctor 有修复结论，
 *  而不是运行到一半才甩卡。
 *
 * 闭环断言：
 * 1. 隔离不可用 → session_start 一次性 warning（含降级后果与
 *    rosclaw doctor 修复入口）；
 * 2. 隔离可用 → 无提示（不噪声）；
 * 3. 默认探测消费 doctor 落盘的 os-isolation.json（探测单源，
 *    §5.3"setup/doctor 一次探测"）；无记录时回落 bwrap 存在性探测。
 */

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

async function buildExtension(
	home: string,
	probe?: () => { isolationReady: boolean },
) {
	const { createRosclawExtension } = await import("../src/extension/index.js");
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const { LocaleManager } = await import("../src/i18n/locale.js");
	const { resolveTaskContext } = await import("../src/native/active-task-context.js");

	const handlers = new Map<string, Handler[]>();
	const pi = {
		on(name: string, handler: Handler) {
			const list = handlers.get(name) ?? [];
			list.push(handler);
			handlers.set(name, list);
		},
		registerCommand() {},
		registerShortcut() {},
		registerEntryRenderer() {},
		registerMessageRenderer() {},
		appendEntry() {},
	};
	const active = new ActiveSessionContext({
		sessionId: "pi_test", missionId: undefined, contextRevision: 0,
		mode: "SIMULATION", profile: "developer", contextState: "LOADING",
		leaseState: "NONE", actionsAllowed: false,
	});
	const call = async () => ({ ok: false, error: "no bridge in test" });
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: home, active,
		leaseManager: new SessionLeaseManager(home, call),
		notify: () => undefined, call,
	});
	const center = new ProductStateCenter({
		rosclawHome: home, active,
		operatorSocket: join(home, "run", "operatord.sock"),
		productVersion: "0.1.0", call: call as never,
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = new LocaleManager(join(home, "agent"));
	const factory = createRosclawExtension({
		profile: "developer", version: "0.1.0", systemPrompt: "TEST",
		active, coordinator, center, locale, rosclawHome: home,
		taskContext: resolveTaskContext({ rosclawHome: home, cwd: "/tmp", mode: "SIMULATION" }),
		...(probe ? { osIsolationProbe: probe } : {}),
	});
	factory(pi as never);
	return handlers;
}

function fakeCtx(notices: string[]) {
	return {
		hasUI: true,
		ui: new Proxy({
			notify: (text: string, _kind?: string) => { notices.push(text); },
		}, {
			get(target, prop) {
				if (prop in target) return (target as Record<PropertyKey, unknown>)[prop];
				return () => undefined; // 其余 ui 方法自动 stub
			},
		}),
	};
}

test("R1-c: 无 OS 沙箱 → session_start 一次性提示（降级后果 + doctor 修复入口）", async () => {
	const home = mkdtempSync(join(tmpdir(), "r1c-"));
	const handlers = await buildExtension(home, () => ({ isolationReady: false }));
	const notices: string[] = [];
	for (const h of handlers.get("session_start") ?? []) await h({}, fakeCtx(notices));
	const hints = notices.filter((n) => n.includes("OS 沙箱"));
	assert.equal(hints.length, 1, `应恰好提示一次，实际 ${hints.length}`);
	assert.match(hints[0], /确认卡|降级/);
	assert.match(hints[0], /rosclaw doctor/);
});

test("R1-c: 隔离可用 → 无提示（不噪声）", async () => {
	const home = mkdtempSync(join(tmpdir(), "r1c-"));
	const handlers = await buildExtension(home, () => ({ isolationReady: true }));
	const notices: string[] = [];
	for (const h of handlers.get("session_start") ?? []) await h({}, fakeCtx(notices));
	assert.equal(notices.filter((n) => n.includes("OS 沙箱")).length, 0);
});

test("R1-c: 默认探测消费 doctor 落盘的 os-isolation.json（单源）", async () => {
	const home = mkdtempSync(join(tmpdir(), "r1c-"));
	mkdirSync(join(home, "agent"), { recursive: true });
	writeFileSync(join(home, "agent", "os-isolation.json"), JSON.stringify({
		checked_at: "2026-09-03T00:00:00Z",
		bwrap: { path: null, smoke_ok: false, detail: "bwrap 未安装" },
		isolation_ready: false,
	}));
	// 不注入 probe——走默认（读 doctor 落盘记录）。
	const handlers = await buildExtension(home);
	const notices: string[] = [];
	for (const h of handlers.get("session_start") ?? []) await h({}, fakeCtx(notices));
	assert.equal(
		notices.filter((n) => n.includes("OS 沙箱")).length, 1,
		"doctor 记录 isolation_ready=false 时必须提示",
	);
});
