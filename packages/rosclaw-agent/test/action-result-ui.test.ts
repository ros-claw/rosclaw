/** Action Result UI 单元测试（五审 P0-5F）：冲突状态机的快速回归。
 *
 * PTY 旅程证明端到端渲染；本文件用多 handler mock pi 验证事件时序：
 * tool_execution_end →(turn_end)→ 助手叙述 message_end → turn_end
 * 的 conflict 条目落盘语义（pi 里 tool 执行属于 tool_call turn，
 * 助手最终回答在下一个 turn——outcome 不能提前清除）。
 */

import assert from "node:assert/strict";
import test from "node:test";

import { createRosclawExtension } from "../src/extension/index.js";

type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

interface AppendedEntry {
	customType: string;
	data: unknown;
}

async function collectHarness() {
	const handlers = new Map<string, Handler[]>();
	const appended: AppendedEntry[] = [];
	const entryRenderers = new Map<string, (entry: { data?: never }, opts: unknown, theme: unknown) => unknown>();
	const pi = {
		on(name: string, handler: Handler) {
			const list = handlers.get(name) ?? [];
			list.push(handler);
			handlers.set(name, list);
		},
		registerCommand() {},
		registerShortcut() {},
		registerEntryRenderer(customType: string, renderer: never) {
			entryRenderers.set(customType, renderer);
		},
		appendEntry(customType: string, data?: unknown) {
			appended.push({ customType, data });
		},
	};
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
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
	const call = async () => ({ ok: false, error: "no bridge in test" });
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: "/tmp/rh-test",
		active,
		leaseManager: new SessionLeaseManager("/tmp/rh-test", call),
		notify: () => undefined,
		call,
	});
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const { LocaleManager } = await import("../src/i18n/locale.js");
	const center = new ProductStateCenter({
		rosclawHome: "/tmp/rh-test",
		active,
		operatorSocket: "/tmp/rh-test/run/operatord.sock",
		productVersion: "0.1.0",
		call: call as never,
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = new LocaleManager("/tmp/rh-test/agent");
	const factory = createRosclawExtension({
		profile: "developer",
		version: "0.1.0",
		systemPrompt: "TEST PROMPT",
		active,
		coordinator,
		center,
		locale,
		rosclawHome: "/tmp/rh-test",
	});
	factory(pi as never);
	const emit = async (name: string, event: unknown) => {
		for (const handler of handlers.get(name) ?? []) {
			await handler(event, {});
		}
	};
	return { emit, appended, entryRenderers };
}

const LIE = "动作已执行，结构化回执已确认。";

test("rejected outcome + 助手谎称完成 → conflict 条目在下一个 turn_end 落盘", async () => {
	const { emit, appended } = await collectHarness();
	// turn 1：toolCall 消息 → tool 执行被拒 → turn_end。
	await emit("message_end", {
		message: { role: "assistant", content: [{ type: "toolCall", name: "rosclaw_request_action" }] },
	});
	await emit("tool_execution_end", {
		toolName: "rosclaw_request_action",
		isError: true,
		result: { details: { status: "REJECTED", capability_id: "", error_code: "INVALID_ARGUMENTS" } },
	});
	await emit("turn_end", { message: { content: [] } });
	// outcome 必须在第一个 turn_end 存活（叙述还没来）。
	// turn 2：助手谎称完成 → turn_end 落 conflict 条目。
	await emit("message_end", { message: { role: "assistant", content: [{ type: "text", text: LIE }] } });
	assert.ok(
		!appended.some((e) => e.customType === "rosclaw.action_conflict"),
		"message_end 内不得落 conflict 条目（会话层会丢）",
	);
	await emit("turn_end", { message: { content: [{ type: "text", text: LIE }] } });
	const results = appended.filter((e) => e.customType === "rosclaw.action_result");
	const conflicts = appended.filter((e) => e.customType === "rosclaw.action_conflict");
	assert.equal(results.length, 1);
	assert.equal((results[0].data as { status: string }).status, "REJECTED");
	assert.equal(conflicts.length, 1, "谎言后必须落 conflict 条目");
	assert.equal((conflicts[0].data as { status: string }).status, "REJECTED");
});

test("COMPLETED outcome + 完成叙述 → 不误报 conflict", async () => {
	const { emit, appended } = await collectHarness();
	await emit("tool_execution_end", {
		toolName: "rosclaw_request_action",
		isError: false,
		result: {
			details: {
				status: "COMPLETED",
				capability_id: "limo.speaker.play_tone",
				approval_id: "appr_1",
				grant_id: "gr_1",
				txn_id: "txn_1",
				action_id: "act_1",
				receipt_id: "rcpt_1",
			},
		},
	});
	await emit("turn_end", { message: { content: [] } });
	await emit("message_end", { message: { role: "assistant", content: [{ type: "text", text: LIE }] } });
	await emit("turn_end", { message: { content: [{ type: "text", text: LIE }] } });
	assert.ok(!appended.some((e) => e.customType === "rosclaw.action_conflict"), "真实完成不得误报");
	const result = appended.find((e) => e.customType === "rosclaw.action_result");
	assert.ok(result, "结果卡条目必须落盘");
	const data = result.data as { txnId?: string; receiptId?: string };
	assert.equal(data.txnId, "txn_1");
	assert.equal(data.receiptId, "rcpt_1");
});

test("被拒后助手诚实报告失败 → 不误报 conflict", async () => {
	const { emit, appended } = await collectHarness();
	await emit("tool_execution_end", {
		toolName: "rosclaw_request_action",
		isError: true,
		result: { details: { status: "REJECTED", capability_id: "x", error_code: "INVALID_ARGUMENTS" } },
	});
	await emit("turn_end", { message: { content: [] } });
	await emit("message_end", {
		message: { role: "assistant", content: [{ type: "text", text: "动作提案被拒，未执行任何物理动作。" }] },
	});
	await emit("turn_end", { message: { content: [] } });
	assert.ok(!appended.some((e) => e.customType === "rosclaw.action_conflict"), "诚实叙述不得误报");
});

test("conflict 渲染器输出内核覆盖文案", async () => {
	const { entryRenderers } = await collectHarness();
	const renderer = entryRenderers.get("rosclaw.action_conflict");
	assert.ok(renderer, "conflict 渲染器必须注册");
	const component = renderer(
		{ data: { claim: LIE, status: "REJECTED" } as never },
		{},
		{},
	) as { render(width: number): string[] };
	const text = component.render(80).join("\n");
	assert.match(text, /冲突/);
	assert.match(text, /未被接受/);
	assert.match(text, /REJECTED/);
});
