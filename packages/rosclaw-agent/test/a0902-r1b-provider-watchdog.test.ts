/** 0902 R1-b 红测试：Provider 分阶段 watchdog（审计 §7）。

0902 实证 + 0901 live 复核实证：Provider 停滞（服务端无声）→
用户看着 Working… 静默 300 秒（pi 默认 httpIdleTimeout）——"Provider
是根因"不等于 ROSClaw 没责任：超时提示、取消、恢复都是产品责任。

分阶段（§7）：
- 首 token 迟滞：10s 提示，30s 取消（取消传播到模型请求——
  pi abort）；
- 流式 idle：15s 状态更新，45s 恢复（取消）；
- 有字节流动的长任务不杀（只在静默时触发——不违反"不杀 turn"
  红线：取消的是无声停滞，不是活跃生成）；
- 回合终态/空闲即解除。
*/

import assert from "node:assert/strict";
import test from "node:test";

test("R1-b: 首 token 10s 无内容 → 提示", async () => {
	const { ProviderStallWatchdog } = await import(
		"../src/native/provider-watchdog.js"
	);
	const notices: string[] = [];
	let aborted = 0;
	const wd = new ProviderStallWatchdog({
		notice: (t) => notices.push(t),
		stallAbort: () => { aborted++; },
		// 测试用快速阈值。
		firstTokenNoticeMs: 50, firstTokenAbortMs: 200,
		streamIdleStatusMs: 100, streamIdleAbortMs: 300,
	});
	wd.turnStarted();
	await new Promise((r) => setTimeout(r, 80));
	assert.equal(notices.length, 1, "10s 无内容未提示");
	assert.match(notices[0], /响应|迟滞|慢/);
	assert.equal(aborted, 0, "提示期就取消了");
	wd.turnEnded();
});

test("R1-b: 内容在 5s 到达 → 不提示不取消", async () => {
	const { ProviderStallWatchdog } = await import(
		"../src/native/provider-watchdog.js"
	);
	const notices: string[] = [];
	let aborted = 0;
	const wd = new ProviderStallWatchdog({
		notice: (t) => notices.push(t),
		stallAbort: () => { aborted++; },
		firstTokenNoticeMs: 50, firstTokenAbortMs: 200,
		streamIdleStatusMs: 100, streamIdleAbortMs: 300,
	});
	wd.turnStarted();
	await new Promise((r) => setTimeout(r, 20));
	wd.contentProgress(); // 5s 内首内容
	await new Promise((r) => setTimeout(r, 20));
	wd.contentProgress();
	await new Promise((r) => setTimeout(r, 120));
	// 流式 idle 状态更新不算取消——aborted 仍 0（150ms < 300ms idle 阈值）。
	assert.equal(aborted, 0);
	wd.turnEnded();
});

test("R1-b: 首 token 30s 无内容 → 取消一次（取消传播）", async () => {
	const { ProviderStallWatchdog } = await import(
		"../src/native/provider-watchdog.js"
	);
	let aborted = 0;
	const wd = new ProviderStallWatchdog({
		notice: () => {},
		stallAbort: () => { aborted++; },
		firstTokenNoticeMs: 30, firstTokenAbortMs: 120,
		streamIdleStatusMs: 1000, streamIdleAbortMs: 3000,
	});
	wd.turnStarted();
	await new Promise((r) => setTimeout(r, 200));
	assert.equal(aborted, 1, "首 token 停滞未取消");
	await new Promise((r) => setTimeout(r, 150));
	assert.equal(aborted, 1, "取消后仍重复取消");
	wd.turnEnded();
});

test("R1-b: 流式中途 idle 45s → 恢复（取消）；活跃流不杀", async () => {
	const { ProviderStallWatchdog } = await import(
		"../src/native/provider-watchdog.js"
	);
	let aborted = 0;
	const wd = new ProviderStallWatchdog({
		notice: () => {},
		stallAbort: () => { aborted++; },
		firstTokenNoticeMs: 30, firstTokenAbortMs: 120,
		streamIdleStatusMs: 60, streamIdleAbortMs: 160,
	});
	wd.turnStarted();
	wd.contentProgress();
	// 活跃流动——每 40ms 一次内容，idle 阈值 160ms 不触发。
	for (let i = 0; i < 3; i++) {
		await new Promise((r) => setTimeout(r, 40));
		wd.contentProgress();
	}
	assert.equal(aborted, 0, "活跃流被杀了（违反不杀 turn 红线）");
	// 然后停滞 → 160ms 后取消。
	await new Promise((r) => setTimeout(r, 220));
	assert.equal(aborted, 1, "流式 idle 未进入恢复");
	wd.turnEnded();
});

test("R1-b: 回合终态后定时器全解除（无 stray abort）", async () => {
	const { ProviderStallWatchdog } = await import(
		"../src/native/provider-watchdog.js"
	);
	let aborted = 0;
	const wd = new ProviderStallWatchdog({
		notice: () => {},
		stallAbort: () => { aborted++; },
		firstTokenNoticeMs: 30, firstTokenAbortMs: 80,
		streamIdleStatusMs: 60, streamIdleAbortMs: 100,
	});
	wd.turnStarted();
	wd.turnEnded();
	await new Promise((r) => setTimeout(r, 200));
	assert.equal(aborted, 0, "回合结束后仍触发取消");
});
