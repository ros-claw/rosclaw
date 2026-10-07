/** PI104 Provider 进展 UX 与看门狗兼容修复测试（SOFTWARE_SOURCE_ONLY）。
 *
 * 实证背景（ACTUAL_PI104_BOUNDARY_COUNTER）：PI1.0.4 会发重复的空
 * text_start 边界事件与空 text_delta——旧 message_update 钩子无条件
 * contentProgress，看门狗被持续重置，首 token/流式 idle 取消永不触发。
 * 修复：只有非空 text/thinking/toolcall 增量才算真实进展；活动区展示
 * 诚实的等待阶段/耗时（不含任何思考文本）。
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
	ProviderStallWatchdog,
	isMeaningfulAssistantEvent,
	providerWatchdogTimingFromEnv,
	type ProviderStallWatchdogOptions,
} from "../src/native/provider-watchdog.js";
import { phaseWorkingMessage } from "../src/extension/activity.js";

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

function makeWatchdog(overrides: Partial<ProviderStallWatchdogOptions> = {}) {
	const notices: string[] = [];
	let aborts = 0;
	const wd = new ProviderStallWatchdog({
		notice: (t) => notices.push(t),
		stallAbort: () => { aborts += 1; },
		firstTokenNoticeMs: 40,
		firstTokenAbortMs: 120,
		streamIdleStatusMs: 40,
		streamIdleAbortMs: 120,
		...overrides,
	});
	return { wd, notices, aborts: () => aborts };
}

test("PI104 classification: empty boundary/empty delta are not progress; nonempty deltas are", () => {
	assert.equal(isMeaningfulAssistantEvent({ type: "text_start" }), false);
	assert.equal(isMeaningfulAssistantEvent({ type: "text_start", delta: "x" }), false);
	assert.equal(isMeaningfulAssistantEvent({ type: "thinking_start" }), false);
	assert.equal(isMeaningfulAssistantEvent({ type: "toolcall_start" }), false);
	assert.equal(isMeaningfulAssistantEvent({ type: "text_end" }), false);
	assert.equal(isMeaningfulAssistantEvent({ type: "text_delta", delta: "" }), false);
	assert.equal(isMeaningfulAssistantEvent({ type: "text_delta", delta: "x" }), true);
	assert.equal(isMeaningfulAssistantEvent({ type: "thinking_delta", delta: "secret" }), true);
	assert.equal(isMeaningfulAssistantEvent({ type: "toolcall_delta", delta: "{}" }), true);
	// 旧 pi 兼容：无结构事件保持旧语义（视为流动）。
	assert.equal(isMeaningfulAssistantEvent(undefined), true);
	assert.equal(isMeaningfulAssistantEvent({ type: "unknown_future" }), true);
});

test("empty text_start stream does not postpone first-token cancellation (PI104 counterexample fixed)", async () => {
	const { wd, aborts } = makeWatchdog();
	wd.turnStarted();
	const started = Date.now();
	while (Date.now() - started < 300) {
		wd.assistantEventProgress({ type: "text_start", delta: "x", contentIndex: 0 });
		await wait(20);
	}
	assert.equal(aborts(), 1, "empty boundary events must not keep the watchdog alive");
	wd.turnEnded();
});

test("empty text_delta stream does not postpone stream-idle cancellation", async () => {
	const { wd, aborts } = makeWatchdog();
	wd.turnStarted();
	wd.assistantEventProgress({ type: "text_delta", delta: "hi" }); // 进入流式
	const started = Date.now();
	while (Date.now() - started < 300) {
		wd.assistantEventProgress({ type: "text_delta", delta: "" });
		await wait(20);
	}
	assert.equal(aborts(), 1, "empty deltas must not renew stream idle");
	wd.turnEnded();
});

test("nonempty text/thinking/toolcall deltas are genuine progress (long streams not killed)", async () => {
	for (const type of ["text_delta", "thinking_delta", "toolcall_delta"]) {
		const { wd, aborts } = makeWatchdog();
		wd.turnStarted();
		const started = Date.now();
		while (Date.now() - started < 300) {
			assert.equal(wd.assistantEventProgress({ type, delta: "x" }), true);
			await wait(20);
		}
		assert.equal(aborts(), 0, `${type} nonempty stream must stay alive`);
		wd.turnEnded();
	}
});

test("honest phase + elapsed: waiting → streaming; tool/user pause excluded from provider wait", async () => {
	const { wd } = makeWatchdog();
	assert.equal(wd.currentPhase(), "idle");
	wd.turnStarted();
	assert.equal(wd.currentPhase(), "waiting");
	await wait(50);
	const waitedBefore = wd.providerWaitElapsedMs();
	assert.ok(waitedBefore >= 40, `expected real wait, got ${waitedBefore}`);
	wd.pauseForTool();
	assert.equal(wd.currentPhase(), "tool");
	const frozen = wd.providerWaitElapsedMs();
	await wait(60);
	assert.ok(Math.abs(wd.providerWaitElapsedMs() - frozen) < 30, "tool time is not provider wait");
	wd.resumeFromTool();
	assert.equal(wd.currentPhase(), "waiting");
	wd.assistantEventProgress({ type: "text_delta", delta: "x" });
	assert.equal(wd.currentPhase(), "streaming");
	wd.pauseForUser();
	assert.equal(wd.currentPhase(), "user_decision");
	const frozen2 = wd.providerWaitElapsedMs();
	await wait(60);
	assert.ok(Math.abs(wd.providerWaitElapsedMs() - frozen2) < 30, "user decision time is not provider wait");
	wd.resumeFromUser();
	wd.turnEnded();
	assert.equal(wd.currentPhase(), "idle");
});

test("activity message shows honest phase + seconds and never thinking text", () => {
	const waiting = phaseWorkingMessage({ currentTool: null, operation: null, provider: { phase: "waiting", elapsedMs: 3200 } });
	assert.match(waiting, /等待模型首个响应/);
	assert.match(waiting, /3\.2s/);
	const streaming = phaseWorkingMessage({ currentTool: null, operation: null, provider: { phase: "streaming", elapsedMs: 1500 } });
	assert.match(streaming, /模型输出中/);
	assert.match(streaming, /1\.5s/);
	const user = phaseWorkingMessage({ currentTool: null, operation: null, provider: { phase: "user_decision", elapsedMs: 800 } });
	assert.match(user, /等待用户确认/);
	assert.match(user, /暂停/);
	// 思考内容绝不泄露：消息里只有阶段与数字。
	for (const msg of [waiting, streaming, user]) assert.ok(!msg.includes("secret") && !msg.includes("思考"));
	// 无 provider 上下文保持旧兜底（n9 既有断言不变）。
	assert.equal(phaseWorkingMessage({ currentTool: null, operation: null }), "Working…");
	assert.equal(phaseWorkingMessage({ currentTool: "bash", operation: null }), "调用 bash");
});

test("env override semantics preserved (thresholds bounded, invalid ignored)", () => {
	const t = providerWatchdogTimingFromEnv({
		ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS: "1000",
		ROSCLAW_PROVIDER_STREAM_IDLE_TIMEOUT_MS: "1000",
	});
	assert.equal(t.firstTokenAbortMs, 1000);
	assert.equal(t.streamIdleAbortMs, 1000);
	const bad = providerWatchdogTimingFromEnv({ ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS: "abc" });
	assert.equal(bad.firstTokenAbortMs, 30_000);
});

test("terminal disarm: no stray abort after turn end (timers disposed)", async () => {
	const { wd, aborts } = makeWatchdog();
	wd.turnStarted();
	wd.assistantEventProgress({ type: "text_delta", delta: "x" });
	wd.turnEnded();
	await wait(250);
	assert.equal(aborts(), 0);
});

test("overlap resume: provider wait resumes only when ALL pauses are clear", async () => {
	// 两种顺序都必须不计算重叠暂停期：user 先恢复（tool 仍忙）与
	// tool 先恢复（user 仍忙）。
	for (const order of ["user_resumes_first", "tool_resumes_first"]) {
		const { wd } = makeWatchdog();
		wd.turnStarted();
		await wait(40);
		wd.pauseForTool();
		wd.pauseForUser();
		if (order === "user_resumes_first") wd.resumeFromUser();
		else wd.resumeFromTool();
		const frozen = wd.providerWaitElapsedMs();
		await wait(90);
		const later = wd.providerWaitElapsedMs();
		assert.ok(later - frozen < 30, `${order}: one pause still active must keep clock paused (grew ${later - frozen}ms)`);
		if (order === "user_resumes_first") wd.resumeFromTool();
		else wd.resumeFromUser();
		await wait(40);
		assert.ok(wd.providerWaitElapsedMs() > later, `${order}: clock resumes after all pauses clear`);
		wd.turnEnded();
	}
});

test("nested tool pauses keep clock paused until the innermost resumes", async () => {
	const { wd } = makeWatchdog();
	wd.turnStarted();
	await wait(30);
	wd.pauseForTool();
	wd.pauseForTool();
	wd.resumeFromTool(); // 还有一层未结束
	const frozen = wd.providerWaitElapsedMs();
	await wait(60);
	assert.ok(wd.providerWaitElapsedMs() - frozen < 30, "nested tool: outer pause still active must not resume clock");
	wd.resumeFromTool();
	await wait(30);
	assert.ok(wd.providerWaitElapsedMs() > frozen, "clock resumes after last tool ends");
	wd.turnEnded();
});

test("terminal freeze: measured elapsed preserved at turn end, new turn resets", async () => {
	const { wd } = makeWatchdog();
	wd.turnStarted();
	await wait(50);
	wd.turnEnded();
	const frozen = wd.providerWaitElapsedMs();
	assert.ok(frozen >= 40, `terminal elapsed must preserve measured wait, got ${frozen}`);
	await wait(80);
	assert.ok(Math.abs(wd.providerWaitElapsedMs() - frozen) < 20, "idle time after terminal must not accumulate");
	wd.turnStarted();
	assert.ok(wd.providerWaitElapsedMs() < 30, "new turn resets its own elapsed");
	wd.turnEnded();
});
