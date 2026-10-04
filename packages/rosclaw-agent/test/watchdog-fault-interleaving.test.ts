import assert from "node:assert/strict";
import test from "node:test";
import { ProviderStallWatchdog } from "../src/native/provider-watchdog.js";

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
function fixture(notice?: () => void, abort?: () => void) {
	const notices: string[] = [];
	let aborts = 0;
	const wd = new ProviderStallWatchdog({
		notice: text => { notices.push(text); notice?.(); },
		stallAbort: () => { aborts++; abort?.(); },
		firstTokenNoticeMs: 10, firstTokenAbortMs: 50,
		streamIdleStatusMs: 10, streamIdleAbortMs: 50,
	});
	return { wd, notices, aborts: () => aborts };
}

test("closing user modal during a tool must not show Provider stall warning", async () => {
	const { wd, notices, aborts } = fixture();
	wd.turnStarted(); wd.contentProgress(); wd.pauseForTool(); wd.pauseForUser();
	wd.resumeFromUser();
	await wait(90);
	assert.equal(notices.length, 0, "tool work incorrectly shows a Provider stall notice");
	assert.equal(aborts(), 0);
	wd.resumeFromTool(); await wait(90);
	assert.equal(aborts(), 1);
});

test("nested tool completion inside user wait must stay paused", async () => {
	const { wd, notices, aborts } = fixture();
	wd.turnStarted(); wd.contentProgress(); wd.pauseForUser(); wd.pauseForTool(); wd.pauseForTool();
	wd.resumeFromTool(); wd.resumeFromTool();
	await wait(90); assert.equal(aborts(), 0); assert.equal(notices.length, 0);
	wd.resumeFromUser(); await wait(90); assert.equal(aborts(), 1);
});

test("notification and abort callback exceptions never escape timer", async () => {
	const { wd, aborts } = fixture(() => { throw Error("UI offline"); }, () => { throw Error("transport offline"); });
	wd.turnStarted(); await wait(90); assert.equal(aborts(), 1);
	wd.resumeFromTool(); await wait(90); assert.equal(aborts(), 1);
});

test("independent watchdog instances cannot cancel each other's active streams", async () => {
	const a = fixture(); const b = fixture(); a.wd.turnStarted(); b.wd.turnStarted();
	for (let i = 0; i < 8; i++) { b.wd.contentProgress(); await wait(15); }
	assert.equal(a.aborts(), 1); assert.equal(b.aborts(), 0);
	b.wd.turnEnded(); await wait(90); assert.equal(b.aborts(), 0);
});

test("end while paused leaves no delayed resume cancellation", async () => {
	const { wd, notices, aborts } = fixture();
	wd.turnStarted(); wd.pauseForTool(); wd.pauseForUser(); wd.turnEnded(); wd.resumeFromUser(); wd.resumeFromTool();
	await wait(90); assert.equal(aborts(), 0); assert.equal(notices.length, 0);
});

for (const mode of ["missing abort API", "abort callback throws"] as const) {
	test(`stall notice describes a cancellation request when ${mode}`, async () => {
		const notices: string[] = [];
		let requests = 0;
		const watchdog = new ProviderStallWatchdog({
			notice: text => notices.push(text),
			stallAbort: () => {
				requests++;
				if (mode === "abort callback throws") throw Error("abort transport unavailable");
				// Contexts without abort can only request manual interruption.
			},
			firstTokenNoticeMs: 10, firstTokenAbortMs: 30,
		});
		watchdog.turnStarted();
		try {
			await wait(80);
			assert.equal(requests, 1, "deadline must still attempt cancellation once");
			const terminal = notices.find(text => text.startsWith("Provider 无响应"));
			assert.ok(terminal, "stall deadline must remain visible");
			assert.ok(!terminal.includes("已取消"), "unconfirmed cancellation cannot be reported as complete");
			assert.ok(terminal.includes("请求取消"));
			await wait(50);
			assert.equal(requests, 1, "callback failure cannot start an abort loop");
		} finally { watchdog.turnEnded(); }
	});
}
