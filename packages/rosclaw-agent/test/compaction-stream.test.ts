import assert from "node:assert/strict";
import test from "node:test";
import { createAssistantMessageEventStream, InMemoryCredentialStore, type AssistantMessage, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager, generateSummaryWithUsage } from "@earendil-works/pi-coding-agent";
import { observeCompactionStream } from "../src/harness/pi/compaction-stream.js";
import type { AssistantMessageEvent } from "@earendil-works/pi-ai";

const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions",
    provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 100 };

function message(): AssistantMessage {
    return { role: "assistant", api: model.api, provider: model.provider, model: model.id,
        content: [{ type: "text", text: "preserve original user goal" }],
        usage: { input: 101, output: 12, cacheRead: 7, cacheWrite: 0, totalTokens: 120,
            cost: { input: .101, output: .024, cacheRead: .007, cacheWrite: 0, total: .132 } },
        stopReason: "stop", timestamp: 1 };
}

test("native PI summary result exposes actual first content and preserves measured usage", async () => {
    const records: Record<string, unknown>[] = [];
    const controller = new AbortController();
    const final = message();
    const streamFn = observeCompactionStream((_model, _context, options) => {
        assert.equal(options?.signal, controller.signal);
        const stream = createAssistantMessageEventStream();
        stream.push({ type: "start", partial: final });
        setTimeout(() => stream.push({ type: "text_delta", contentIndex: 0, delta: "preserve", partial: final }), 15);
        setTimeout(() => stream.push({ type: "done", reason: "stop", message: final }), 30);
        return stream;
    }, { enabled: () => true, record: record => records.push(record), waitingNoticeMs: 5 });
    const result = await generateSummaryWithUsage([], model, 100, undefined, undefined,
        controller.signal, undefined, undefined, undefined, streamFn);
    assert.equal(result.text, "preserve original user goal");
    assert.deepEqual(result.usage, final.usage);
    assert.ok(records.some(record => record.status === "first_content"));
    assert.ok(records.some(record => record.status === "stream_completed"));
    assert.equal(controller.signal.aborted, false);
});

test("all provider events retain exact order and object identity", async () => {
    const final = message();
    const frames: AssistantMessageEvent[] = [
        { type: "start", partial: final },
        { type: "thinking_delta", contentIndex: 0, delta: "reason", partial: final },
        { type: "text_delta", contentIndex: 0, delta: "preserve", partial: final },
        { type: "done", reason: "stop", message: final },
    ];
    const native = createAssistantMessageEventStream();
    for (const frame of frames) native.push(frame);
    const observed = await observeCompactionStream(() => native, {
        enabled: () => true, record: () => { throw Error("disk full"); },
    })(model, {} as never);
    const actual = [];
    for await (const event of observed) actual.push(event);
    assert.equal(actual.length, frames.length);
    actual.forEach((event, index) => assert.equal(event, frames[index]));
    assert.equal(await observed.result(), final);
});

test("public result-only stream termination is not misclassified as a missing terminal", async () => {
    const final = message();
    const records: Record<string, unknown>[] = [];
    const source = createAssistantMessageEventStream(); source.end(final);
    const output = await observeCompactionStream(() => source, {
        enabled: () => true, record: record => records.push(record),
    })(model, {} as never);
    assert.equal(await output.result(), final);
    assert.equal(records.at(-1)?.terminal_source, "result_only");
    assert.equal(records.at(-1)?.status, "stream_completed");
    const events = [];
    for await (const event of output) events.push(event);
    assert.deepEqual(events, []);
});

for (const reason of ["error", "aborted"] as const) {
    test(`actual ${reason} result and usage remain unchanged`, async () => {
        const final = { ...message(), stopReason: reason, errorMessage: "provider fixture error" };
        const source = createAssistantMessageEventStream();
        const event: AssistantMessageEvent = { type: "error", reason, error: final };
        source.push(event);
        const records: Record<string, unknown>[] = [];
        const output = await observeCompactionStream(() => source, {
            enabled: () => true, record: record => records.push(record),
        })(model, {} as never);
        assert.equal(await output.result(), final);
        const first = await output[Symbol.asyncIterator]().next();
        assert.equal(first.value, event);
        assert.equal(records.at(-1)?.status, reason === "aborted" ? "stream_cancelled" : "stream_failed");
        assert.equal(records.at(-1)?.usage, final.usage);
    });
}

for (const fault of ["throws", "rejects", "missing_terminal"] as const) {
    test(`native summary fails closed for malformed source ${fault}`, async () => {
        const records: Record<string, unknown>[] = [];
        const wrapped = observeCompactionStream(() => {
            if (fault === "throws") throw Error("source threw");
            if (fault === "rejects") return Promise.reject(Error("source rejected"));
            const source = createAssistantMessageEventStream(); source.end(); return source;
        }, { enabled: () => true, record: record => records.push(record) });
        await assert.rejects(generateSummaryWithUsage([], model, 100, undefined, undefined,
            undefined, undefined, undefined, undefined, wrapped), /ROSCLAW_STREAM_PROTOCOL_ERROR/);
        assert.equal(records.at(-1)?.status, "stream_failed");
        assert.equal(records.at(-1)?.usage_status, "NOT_RECORDED");
    });
}

test("slow active summary and no-content waiting are observed without speculative cancellation", async () => {
    const controller = new AbortController();
    const source = createAssistantMessageEventStream();
    const records: Record<string, unknown>[] = [];
    const output = await observeCompactionStream(() => source, {
        enabled: () => true, record: record => records.push(record), waitingNoticeMs: 5, progressRecordMs: 5,
    })(model, {} as never, { signal: controller.signal });
    await new Promise(resolve => setTimeout(resolve, 18));
    assert.ok(records.some(record => record.status === "waiting" && record.phase === "before_first_content"));
    assert.ok(!records.some(record => record.status === "stream_completed"));
    for (let index = 0; index < 4; index++) {
        source.push({ type: "thinking_delta", contentIndex: 0, delta: "active", partial: message() });
        await new Promise(resolve => setTimeout(resolve, 12));
    }
    assert.ok(records.some(record => record.status === "content_progress"));
    assert.equal(controller.signal.aborted, false);
    const final = message(); source.push({ type: "done", reason: "stop", message: final });
    assert.equal(await output.result(), final);
    const count = records.length;
    await new Promise(resolve => setTimeout(resolve, 15));
    assert.equal(records.length, count);
});

test("disabled observation preserves the original stream and every request argument", async () => {
    const source = createAssistantMessageEventStream(); source.end(message());
    const context = {} as never;
    const options = { signal: new AbortController().signal };
    const wrapped = observeCompactionStream((m, c, o) => {
        assert.equal(m, model); assert.equal(c, context); assert.equal(o, options);
        return source;
    }, { enabled: () => false, record: () => assert.fail("normal turn was observed") });
    assert.equal(wrapped(model, context, options), source);
});

// Real SDK sessions use only in-memory history and a temporary empty agentDir.
// Fixtures exercise the public entry points without network or production credentials.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

for (const mode of ["complete", "abort", "closed", "provider_error"] as const) {
    test(`real AgentSession.compact public stream observes ${mode} with canonical history`, { timeout: 10_000 }, async () => {
        const home = mkdtempSync(join(tmpdir(), "rosclaw-summary-fixture-"));
        const settings = SettingsManager.inMemory({ compaction: { enabled: true,
            reserveTokens: 100, keepRecentTokens: 32 }, retry: { enabled: false } });
        const history = SessionManager.inMemory(home);
        for (let i = 0; i < 6; i++) {
            history.appendMessage({ role: "user", content: "Preserve goal " + "context ".repeat(200), timestamp: i * 2 });
            history.appendMessage({ ...message(), timestamp: i * 2 + 1 });
        }
        const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(),
            modelsPath: null, refreshOnCreate: false });
        runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl,
            apiKey: "non-secret-fixture-key", headers: { "X-Fixture-Route": "retained" }, models: [model] });
        const actualModel = runtime.getModel(model.provider, model.id)!;
        const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
            noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true });
        await loader.reload();
        const { session } = await createAgentSession({ cwd: home, agentDir: home, model: actualModel,
            modelRuntime: runtime, sessionManager: history, settingsManager: settings, resourceLoader: loader,
            noTools: "all", thinkingLevel: "off" });
        const records: Record<string, unknown>[] = [];
        const lifecycle: string[] = [];
        session.subscribe(event => lifecycle.push(event.type));
        let called!: () => void;
        const started = new Promise<void>(resolve => { called = resolve; });
        let signal: AbortSignal | undefined;
        session.agent.streamFunction = observeCompactionStream((requestModel, context, options) => {
            assert.equal(requestModel.provider, actualModel.provider);
            assert.equal(requestModel.id, actualModel.id);
            assert.equal(options?.apiKey, "non-secret-fixture-key");
            assert.equal(options?.headers?.["X-Fixture-Route"], "retained");
            assert.ok(context.messages.length > 0);
            assert.equal(session.isCompacting, true);
            signal = options?.signal;
            assert.ok(signal);
            const source = createAssistantMessageEventStream();
            if (mode === "complete") {
                source.push({ type: "text_delta", contentIndex: 0, delta: "preserve", partial: message() });
                source.end(message());
            } else if (mode === "closed") source.end();
            else if (mode === "provider_error") {
                source.push({ type: "error", reason: "error", error: { ...message(),
                    stopReason: "error", errorMessage: "fixture auth declined" } });
            }
            else signal!.addEventListener("abort", () => {
                const cancelled = { ...message(), stopReason: "aborted" as const };
                source.push({ type: "error", reason: "aborted", error: cancelled });
            }, { once: true });
            called();
            return source;
        }, { enabled: () => session.isCompacting, record: record => records.push(record),
            owner: () => ({ session_id: session.sessionId }) });
        const before = history.getEntries();
        try {
            const compact = session.compact("Retain original goal and receipts");
            await started;
            if (mode === "abort") session.abortCompaction();
            if (mode === "complete") {
                const result = await compact;
                assert.equal(result.summary, "preserve original user goal");
                assert.deepEqual(result.usage, message().usage);
                assert.equal(history.getEntries().filter(entry => entry.type === "compaction").length, 1);
                assert.equal(records.at(-1)?.terminal_source, "result_only");
            } else {
                await assert.rejects(compact, mode === "closed" ? /ROSCLAW_STREAM_PROTOCOL_ERROR/
                    : mode === "provider_error" ? /fixture auth declined/ : /cancelled|aborted/i);
                assert.equal(history.getEntries().filter(entry => entry.type === "compaction").length, 0);
                assert.equal(records.at(-1)?.status, mode === "abort" ? "stream_cancelled" : "stream_failed");
            }
            assert.deepEqual(history.getEntries().slice(0, before.length), before);
            assert.equal(signal?.aborted, mode === "abort");
            assert.equal(session.isCompacting, false);
            assert.ok(lifecycle.includes("compaction_start"));
            assert.ok(lifecycle.includes("compaction_end"));
            assert.ok(records.every(record => record.session_id === session.sessionId));
        } finally { session.dispose(); rmSync(home, { recursive: true, force: true }); }
    });
}


test("enabled observation preserves request receiver, auth, signal and original options object", async () => {
    const receiver = { fixture: true };
    const context = { systemPrompt: "canonical context", messages: [] };
    const options = { apiKey: "fixture-key", headers: { "X-Fixture": "yes" },
        signal: new AbortController().signal, maxTokens: 83 };
    const source = createAssistantMessageEventStream(); source.end(message());
    const wrapped = observeCompactionStream(function(this: unknown, m, c, o) {
        assert.equal(this, receiver); assert.equal(m, model); assert.equal(c, context); assert.equal(o, options);
        return source;
    }, { enabled: () => true, record: () => {}, owner: () => { throw Error("observer unavailable"); } });
    const output = await wrapped.call(receiver, model, context as never, options);
    assert.equal(await output.result(), await source.result());
    assert.equal(options.signal.aborted, false);
});
