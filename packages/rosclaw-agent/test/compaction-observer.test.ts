import assert from "node:assert/strict";
import test from "node:test";
import { createAssistantMessageEventStream, type AssistantMessage, type Model } from "@earendil-works/pi-ai";
import { generateSummaryWithUsage } from "@earendil-works/pi-coding-agent";
import { CompactionObserver } from "../src/native/compaction-observer.js";

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
function fixture() {
	const notices: string[] = [];
	const logs: Record<string, unknown>[] = [];
	const observer = new CompactionObserver({ notice: t => notices.push(t), log: r => logs.push(r),
		firstNoticeMs: 10, repeatNoticeMs: 25 });
	return { observer, notices, logs };
}

test("long native PI summarization stays active despite no session token events", async () => {
	const { observer, notices, logs } = fixture();
	const controller = new AbortController();
	observer.started("threshold", controller.signal);
	const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions",
		provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 100 };
	const summary = await generateSummaryWithUsage([], model, 100, undefined, undefined,
		controller.signal, undefined, undefined, undefined, (_model, _context, options) => {
			assert.equal(options?.signal, controller.signal);
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = { role: "assistant", api: model.api, provider: model.provider,
				model: model.id, content: [{ type: "text", text: "keep user goal and artifact refs" }],
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
				stopReason: "stop", timestamp: 0 };
			setTimeout(() => stream.push({ type: "done", reason: "stop", message }), 75);
			return stream;
		});
	assert.match(summary.text, /user goal/);
	assert.equal(controller.signal.aborted, false);
	assert.ok(notices.length >= 1);
	assert.ok(notices.every(text => /尚无法判断/.test(text)));
	observer.ended("completed");
	const count = logs.length;
	await wait(40);
	assert.equal(logs.length, count);
	assert.equal(logs.at(-1)?.status, "completed");
});

test("abort observer detaches timers and distinguishes cancellation from failure", async () => {
	const { observer, logs, notices } = fixture();
	const controller = new AbortController();
	observer.started("manual", controller.signal);
	controller.abort();
	observer.ended("failed", "late event must not rewrite cancellation");
	await wait(50);
	assert.equal(notices.length, 0);
	assert.deepEqual(logs.map(r => r.status), ["started", "cancelled"]);
	const alreadyAborted = new AbortController(); alreadyAborted.abort();
	observer.started("overflow", alreadyAborted.signal);
	assert.equal(logs.at(-1)?.status, "cancelled");
});

test("independent session observers retain their own deadlines and terminal state", async () => {
	const a = fixture(); const b = fixture();
	a.observer.started("threshold", new AbortController().signal);
	b.observer.started("manual", new AbortController().signal);
	a.observer.ended("failed", "provider disconnected");
	await wait(55);
	assert.equal(a.notices.length, 0);
	assert.ok(b.notices.length >= 1);
	assert.equal(a.logs.at(-1)?.status, "failed");
	b.observer.ended("shutdown");
	assert.equal(b.logs.at(-1)?.status, "shutdown");
});

test("failed logging and UI callbacks never cancel or crash compaction", async () => {
	const controller = new AbortController();
	const observer = new CompactionObserver({ notice: () => { throw Error("UI gone"); },
		log: () => { throw Error("disk full"); }, firstNoticeMs: 5, repeatNoticeMs: 10 });
	observer.started("manual", controller.signal);
	await wait(30);
	assert.equal(controller.signal.aborted, false);
	observer.ended("completed");
});
