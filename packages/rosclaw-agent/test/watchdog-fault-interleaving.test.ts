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
