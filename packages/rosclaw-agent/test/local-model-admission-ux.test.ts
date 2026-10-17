/** Source-authored only: Root must review/compile/run. No live provider or hardware. */
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { classifyAssistantFailure, classifyModelError, ProviderErrorGate } from "../src/native/model-errors.js";
import { mapPiEvent } from "../src/harness/pi/pi-session-adapter.js";
import { ModelRequestAdmission, type ModelRequestAdmissionPolicy } from "../src/harness/pi/model-request-admission.js";
import { installModelRequestAdmission, modelAdmissionSettings } from "../src/harness/pi/pi-model-runtime.js";
import type { HarnessEvent } from "../src/harness/port.js";

const localCodes = ["MODEL_REQUEST_ALLOWANCE_EXHAUSTED", "MODEL_REQUEST_USAGE_UNKNOWN"] as const;
function assertLocalText(text: string) {
    assert.match(text, /本地/);
    assert.match(text, /后续模型调用已阻止/);
    assert.match(text, /已保存进度/);
    assert.match(text, /未完成的交付仍未完成/);
    assert.match(text, /不能保证所有编辑均已保存/);
    assert.match(text, /外部/);
    assert.doesNotMatch(text, /\/model|\/budget|立即继续|自动重试中|任务保持可恢复/);
}

for (const code of localCodes) {
    test(`local typed refusal: ${code}`, () => {
        // Error.name prefix is the SDK's ordinary Error-to-string shape.
        const raw = `ModelRequestAdmissionError: ${code}`;
        const classified = classifyAssistantFailure({ role: "assistant", stopReason: "error", errorMessage: raw })!;
        assert.equal(classified.code, code);
        assert.equal(classified.taskRecoverable, false);
        const gate = new ProviderErrorGate();
        for (const active of [false, true]) {
            gate.onModelSwitch();
            const card = gate.onError(classified, { hasActiveTask: active, raw });
            assert.equal(card.showCard, true);
            assertLocalText(card.cardText);
            assert.equal(card.activity?.code, code);
            assert.equal(gate.onError(classified, { hasActiveTask: active }).showCard, false);
        }
        const failed = mapPiEvent({ type: "message_end", turnId: "t", message: { role: "assistant", stopReason: "error", errorMessage: raw } });
        assert.equal(failed?.type, "turn.failed");
        if (failed?.type !== "turn.failed") throw Error("missing typed failure");
        assert.equal(failed.error.code, code);
        assert.equal(failed.error.retryable, false);
        assertLocalText(failed.error.message);
        assert.deepEqual(mapPiEvent({ type: "agent_end" }), { type: "session.idle" });
    });
}

test("local tokens win over incidental provider words, not over authoritative cancellation/length", () => {
    for (const code of localCodes) {
        assert.equal(classifyModelError(`${code}: quota 403 network timeout`).code, code);
        assert.equal(classifyAssistantFailure({ role: "assistant", stopReason: "aborted", errorMessage: code })?.code, "MODEL_REQUEST_CANCELLED");
        assert.equal(classifyAssistantFailure({ role: "assistant", stopReason: "length", errorMessage: code })?.code, "MODEL_OUTPUT_LIMIT");
        assert.equal(mapPiEvent({ type: "message_end", message: { role: "assistant", stopReason: "aborted", errorMessage: code } })?.type, "turn.cancelled");
    }
    assert.equal(classifyModelError("MODEL_REQUEST_STREAM_USAGE_UNKNOWN").code, "MODEL_REQUEST_USAGE_UNKNOWN");
    assert.equal(classifyModelError("NOT_MODEL_REQUEST_USAGE_UNKNOWN_SUFFIX").code, "MODEL_UNKNOWN");
    const length = mapPiEvent({ type: "message_end", message: { role: "assistant", stopReason: "length" } });
    assert.equal(length?.type, "turn.failed");
    if (length?.type === "turn.failed") {
        assert.equal(length.error.code, "MODEL_OUTPUT_LIMIT");
        assert.equal(length.error.retryable, false);
    }
});

test("real provider controls retain classifications and recovery gate behavior", () => {
    for (const [raw, code] of [
        ["403 usage limit for billing cycle", "PROVIDER_QUOTA_EXHAUSTED"],
        ["401 unauthorized", "MODEL_CREDENTIAL_INVALID"],
        ["429 too many requests", "PROVIDER_RATE_LIMITED"],
        ["fetch failed ECONNRESET", "PROVIDER_UNAVAILABLE"],
        ["WebSocket closed 1012", "PROVIDER_UNAVAILABLE"],
    ]) {
        const classified = classifyModelError(raw);
        assert.equal(classified.code, code);
        const gate = new ProviderErrorGate();
        assert.match(gate.onError(classified, { hasActiveTask: true }).cardText, /\/model/);
        assert.equal(gate.onError(classified, { hasActiveTask: true }).showCard, false);
        gate.onModelSwitch();
        assert.equal(gate.onError(classified, { hasActiveTask: true }).showCard, true);
        const event = mapPiEvent({ type: "message_end", message: { role: "assistant", stopReason: "error", errorMessage: raw } });
        assert.equal(event?.type, "turn.failed");
        if (event?.type === "turn.failed") {
            assert.equal(event.error.code, code);
            assert.equal(event.error.retryable, true);
        }
        gate.onError(classifyModelError("Operation aborted"), { hasActiveTask: true });
        assert.equal(gate.pausedCode, null);
    }
});

// Capture actual registration callbacks without starting lifecycle hooks, timers,
// bridge RPC, resource discovery, or model calls. Only the model-error callback
// is invoked; unrelated mirror/tool hooks are deliberately not event-emitted.
test("registered extension hook displays local refusal without provider-paused or dispatch", { timeout: 10000 }, async () => {
    const { createRosclawExtension } = await import("../src/extension/index.js");
    const home = await mkdtemp(join(tmpdir(), "local-ux-hooks-"));
    try {
        const hooks = new Map<string, Array<(...args: any[]) => any>>();
        const entries: any[] = [], notices: string[] = [], paused: string[] = [];
        let providerOk = 0;
        const pi = new Proxy({
            on(name: string, callback: (...args: any[]) => any) {
                const list = hooks.get(name) ?? []; list.push(callback); hooks.set(name, list);
            },
            appendEntry: (...args: any[]) => entries.push(args),
            sendMessage: () => { throw Error("explanation must not send a message/turn"); },
            sendUserMessage: () => { throw Error("no extra model request"); },
        }, { get(target, key) { return Reflect.get(target, key) ?? (() => undefined); } });
        const options = {
            profile: "developer", version: "inert", systemPrompt: "inert", rosclawHome: home,
            active: { current: { sessionId: "", missionId: "" } }, coordinator: {},
            center: {
                noteWorkspace() {}, noteProviderOk() { providerOk++; },
                noteProviderPaused(code: string) { paused.push(code); },
                // ActiveTask lookup may query; no actual bridge is installed.
                async call() { return { ok: true, tasks: [] }; },
            },
            locale: { effective: "zh-CN" }, workspaceStore: {},
            taskContext: { workspaceRoot: home, workspaceSource: "default" },
        };
        const factory = createRosclawExtension(options as any);
        await factory(pi as any);
        const messageHooks = hooks.get("message_end")!;
        assert.equal(messageHooks.length, 4, "review registration order if extension hooks change");
        const hook = messageHooks[1]; // production classifier/gate callback, not a copied helper
        const ctx = { hasUI: true, ui: { notify(text: string) { notices.push(text); } } };
        for (const code of localCodes) {
            const event = { message: { role: "assistant", stopReason: "error", errorMessage: code } };
            await hook(event, ctx);
            await hook(event, ctx);
        }
        assert.equal(notices.length, 2, "dedupe applies to actual hook");
        for (const notice of notices) assertLocalText(notice);
        assert.deepEqual(paused, []);
        assert.equal(providerOk, 2);
        assert.deepEqual(entries.map(e => e[1].code), [...localCodes]);
        await hook({ message: { role: "assistant", stopReason: "error", errorMessage: "fetch failed" } }, ctx);
        assert.deepEqual(paused, ["PROVIDER_UNAVAILABLE"]);
    } finally { await rm(home, { recursive: true, force: true }); }
});

const policy: ModelRequestAdmissionPolicy = {
    version: 1, lifetime: "runtime", guarantee: "local_reservation",
    inclusiveInputLimit: 100, outputLimit: 100, requestReserveInput: 80, requestReserveOutput: 10,
};

for (const unknown of [false, true]) {
    test(`actual offline SDK error/end/idle; same-gate model switch cannot rescue ${unknown ? "UNKNOWN" : "exhaustion"}`,
        { timeout: 15000 }, async () => {
        const { createAssistantMessageEventStream, InMemoryCredentialStore } = await import("@earendil-works/pi-ai");
        const { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } = await import("@earendil-works/pi-coding-agent");
        const home = await mkdtemp(join(tmpdir(), "local-ux-sdk-"));
        let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
        try {
            const model = {
                id: "offline", name: "offline", api: "openai-completions" as const, provider: "fixture",
                baseUrl: "http://invalid.local", reasoning: false, input: ["text" as const],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 100,
            };
            const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
                modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
            runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "INERT",
                models: [model, { ...model, id: "offline-second" }] });
            const settings = modelAdmissionSettings(SettingsManager.inMemory({ compaction: { enabled: false } }));
            const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
                noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true });
            await loader.reload();
            ({ session } = await createAgentSession({ cwd: home, agentDir: home, modelRuntime: runtime,
                model: runtime.getModel(model.provider, model.id)!, settingsManager: settings, resourceLoader: loader,
                sessionManager: SessionManager.inMemory(home), tools: [], thinkingLevel: "off" }));
            let requests = 0;
            session.agent.streamFunction = () => {
                assert.equal(++requests, 1, "a second underlying dispatch is a regression");
                const stream = createAssistantMessageEventStream();
                const msg = {
                    role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
                    content: [], timestamp: 1, stopReason: unknown ? "error" as const : "stop" as const,
                    usage: { input: 21, cacheRead: 0, cacheWrite: 0, output: 5, totalTokens: 26,
                        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                    ...(unknown ? { errorMessage: "INERT_USAGE_UNAVAILABLE" } : {}),
                };
                if (unknown) stream.push({ type: "error", reason: "error", error: msg });
                else stream.push({ type: "done", reason: "stop", message: msg });
                stream.end();
                return stream;
            };
            const admission = new ModelRequestAdmission(policy);
            installModelRequestAdmission(session, admission);
            const events: HarnessEvent[] = [];
            session.subscribe(event => {
                const mapped = mapPiEvent(event as any);
                if (mapped) events.push(mapped);
            });
            await session.bindExtensions({});
            await session.prompt("one finite inert request");
            assert.equal(requests, 1);
            const code = unknown ? localCodes[1] : localCodes[0];
            const gate = new ProviderErrorGate();
            for (const switched of [false, true]) {
                if (switched) {
                    await session.setModel(runtime.getModel(model.provider, "offline-second")!);
                    gate.onModelSwitch(); // display reset cannot affect the real admission gate
                }
                events.length = 0;
                await session.prompt("denied, not a provider request");
                assert.equal(requests, 1);
                assert.equal(session.isIdle, true);
                const failed = events.find(e => e.type === "turn.failed");
                assert.equal(failed?.type, "turn.failed");
                if (failed?.type !== "turn.failed") throw Error("SDK failed to publish denial");
                assert.equal(failed.error.code, code);
                assert.equal(failed.error.retryable, false);
                assertLocalText(failed.error.message);
                assert.ok(events.findIndex(e => e.type === "session.idle") > events.indexOf(failed));
                assert.equal(events.some(e => e.type === "assistant.completed"), false);
                const last = session.messages.at(-1) as { role: string; stopReason: string; errorMessage: string };
                assert.equal(last.stopReason, "error");
                assertLocalText(gate.onError(classifyAssistantFailure(last)!, { hasActiveTask: true }).cardText);
                assert.equal(requests, 1, "classification/presentation did not dispatch an explanation");
            }
            assert.equal(admission.snapshot().dispatchedRequests, 1);
            assert.equal(admission.snapshot().pendingRequests, 0);
            assert.equal(admission.snapshot().unknownUsageRequests, unknown ? 1 : 0);
            assert.equal(admission.snapshot().knownInclusiveInput, 21);
            assert.equal(admission.snapshot().providerCostBoundVerified, false);
        } finally {
            await session?.abort();
            session?.dispose();
            await rm(home, { recursive: true, force: true });
        }
    });
}

