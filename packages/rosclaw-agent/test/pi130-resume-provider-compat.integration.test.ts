import test from "node:test";
import assert from "node:assert/strict";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "typebox";
import {
    createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager,
    type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import {
    createAssistantMessageEventStream, getCurrentSystemMessage, InMemoryCredentialStore,
    type Api, type AssistantMessage, type Context, type Model, type Usage,
} from "@earendil-works/pi-ai";

// Source-authored for a later Root run on the reviewed/instrumented 1.3 SDK.
// Sessions are genuine; session model streams and provider fetch below are synthetic.
// No worker, socket, credentials from disk, or production transport is required.
type Session = Awaited<ReturnType<typeof createAgentSession>>["session"];
type SessionOptions = NonNullable<Parameters<typeof createAgentSession>[0]>;
const agentRoot = resolve(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"))), "..");
const aiRoot = resolve(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))), "..");
const names = (session: Session) => session.getActiveToolNames().sort();
const syntheticUsage: Usage = {
    input: 7, output: 3, cacheRead: 2, cacheWrite: 0, totalTokens: 12,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
function response(model: Model<Api>, content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"] = "stop"): AssistantMessage {
    return { role: "assistant", api: model.api, provider: model.provider, model: model.id,
        content, stopReason, usage: structuredClone(syntheticUsage), timestamp: Date.now() };
}
function terminal(message: AssistantMessage) {
    const stream = createAssistantMessageEventStream();
    if (message.stopReason === "error" || message.stopReason === "aborted") {
        stream.push({ type: "error", reason: message.stopReason, error: message });
    } else {
        assert.ok(message.stopReason === "stop" || message.stopReason === "toolUse");
        stream.push({ type: "start", partial: message });
        stream.push({ type: "done", reason: message.stopReason, message });
    }
    return stream;
}
function object(value: unknown): asserts value is Record<string, unknown> {
    assert.ok(value !== null && typeof value === "object" && !Array.isArray(value));
}
function json(path: string): unknown { return JSON.parse(readFileSync(path, "utf8")); }
function pin130() {
    for (const root of [agentRoot, aiRoot]) {
        const manifest = json(join(root, "package.json"));
        object(manifest);
        assert.equal(manifest.version, "1.3.0", "never qualify the old SDK with new test names");
    }
}
async function fixture() {
    pin130();
    const cwd = mkdtempSync(join(tmpdir(), "rosclaw-pi130-resume-"));
    const sessions: Session[] = [];
    const calls = { alpha: 0, blocked: 0, vanished: 0 };
    const tool = (name: keyof typeof calls): ToolDefinition => ({
        name, label: name, description: `Synthetic offline ${name}`,
        parameters: Type.Object({}),
        execute: async () => {
            calls[name]++;
            return { content: [{ type: "text", text: `${name} fixture result` }], details: {} };
        },
    });
    const model: Model<"openai-completions"> = {
        id: "resume-fixture", name: "Offline resume", api: "openai-completions", provider: "pi130-fixture",
        baseUrl: "https://invalid.example", reasoning: false, input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 128,
    };
    try {
        const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
            modelsStorePath: join(cwd, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
        runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "SYNTHETIC_OFFLINE", models: [model] });
        const selected = runtime.getModel(model.provider, model.id);
        assert.ok(selected);
        async function create(manager: SessionManager, options: Pick<SessionOptions, "tools" | "noTools" | "excludeTools"> = {}, customTools = [tool("alpha"), tool("blocked")]) {
            const settings = SettingsManager.inMemory({ defaultTools: ["read"], cacheWarming: "off",
                compaction: { enabled: false }, retry: { enabled: false, provider: { maxRetries: 0 } } });
            const loader = new DefaultResourceLoader({ cwd, agentDir: cwd, settingsManager: settings,
                noExtensions: true, noContextFiles: true, noSkills: true, noPromptTemplates: true, noThemes: true });
            await loader.reload();
            assert.deepEqual(loader.getExtensions().errors, []);
            const { session } = await createAgentSession({ cwd, agentDir: cwd, modelRuntime: runtime,
                model: selected, thinkingLevel: "off", settingsManager: settings, resourceLoader: loader,
                sessionManager: manager, customTools, ...options });
            sessions.push(session);
            await session.bindExtensions({});
            // Fail closed if an unplanned session request occurs; replace explicitly per case.
            session.agent.streamFunction = () => { throw new Error("UNPLANNED_PI130_FIXTURE_REQUEST"); };
            return session;
        }
        const original = await create(SessionManager.create(cwd, join(cwd, "saved")), {}, [tool("alpha"), tool("blocked"), tool("vanished")]);
        original.setActiveToolsByName(["alpha", "blocked", "vanished"]);
        original.agent.streamFunction = (actual, context) => {
            assert.deepEqual(getCurrentSystemMessage(context.messages)?.toolsAdded?.map(t => t.name).sort(), ["alpha", "blocked", "vanished"]);
            return terminal(response(actual, [{ type: "text", text: "persist loadout" }]));
        };
        await original.prompt("Persist this actual SDK loadout");
        assert.equal(original.isIdle, true);
        const path = original.sessionFile;
        assert.ok(path && existsSync(path), "real prompt must persist a JSONL session");
        const savedBytes = readFileSync(path);
        const savedId = original.sessionId;
        let copies = 0;
        const snapshot = (source = path) => {
            const target = join(cwd, `resume-${++copies}.jsonl`);
            copyFileSync(source, target);
            const manager = SessionManager.open(target, join(cwd, "resumed"), cwd);
            assert.equal(manager.getSessionId(), savedId);
            return manager;
        };
        const persisted = snapshot();
        assert.deepEqual(getCurrentSystemMessage(persisted.buildSessionContext().messages)?.toolsAdded?.map(t => t.name).sort(), ["alpha", "blocked", "vanished"]);
        return { cwd, create, snapshot, calls, tool, savedId,
            close: () => {
                try { assert.deepEqual(readFileSync(path), savedBytes, "resume variants never rewrite the baseline"); }
                finally { for (const session of sessions) session.dispose(); rmSync(cwd, { recursive: true, force: true }); }
            } };
    } catch (error) {
        for (const session of sessions) session.dispose();
        rmSync(cwd, { recursive: true, force: true });
        throw error;
    }
}

// This would fail if constructor restoration were removed: defaults include read,
// whereas the persisted transcript selected only alpha/blocked/vanished.
test("1.3 default resume restores persisted tools, not denied or unregistered capabilities", { timeout: 15000 }, async () => {
    const f = await fixture();
    try {
        const session = await f.create(f.snapshot(), { excludeTools: ["blocked"] });
        assert.equal(session.sessionId, f.savedId);
        assert.deepEqual(names(session), ["alpha"]);
        assert.equal(session.getToolDefinition("blocked"), undefined);
        assert.equal(session.getToolDefinition("vanished"), undefined, "transcript is not a tool registration");
        session.setActiveToolsByName(["alpha", "blocked", "vanished", "unknown"]);
        assert.deepEqual(names(session), ["alpha"], "public selection cannot re-register denied or missing tools");
        let requests = 0;
        session.agent.streamFunction = (actual, context) => {
            assert.deepEqual(getCurrentSystemMessage(context.messages)?.toolsAdded?.map(t => t.name), ["alpha"]);
            requests++;
            assert.ok(requests <= 3);
            if (requests === 1) return terminal(response(actual, [
                { type: "toolCall", id: "deny-blocked", name: "blocked", arguments: {} },
                { type: "toolCall", id: "deny-vanished", name: "vanished", arguments: {} },
            ], "toolUse"));
            if (requests === 2) return terminal(response(actual, [{ type: "toolCall", id: "allow-alpha", name: "alpha", arguments: {} }], "toolUse"));
            return terminal(response(actual, [{ type: "text", text: "done" }]));
        };
        await session.prompt("Exercise genuine SDK dispatch restrictions");
        assert.equal(requests, 3);
        assert.deepEqual(f.calls, { alpha: 1, blocked: 0, vanished: 0 });
        const results = session.messages.filter(m => m.role === "toolResult");
        assert.equal(results.length, 3);
        assert.deepEqual(results.map(m => [m.toolCallId, m.isError]), [["deny-blocked", true], ["deny-vanished", true], ["allow-alpha", false]]);
        assert.equal(session.isIdle, true);
        assert.deepEqual(names(session), ["alpha"]);
    } finally { f.close(); }
});

test("1.3 explicit allowlist, empty selection, modifiers and fresh-session boundaries", { timeout: 15000 }, async () => {
    const f = await fixture();
    try {
        const explicit = await f.create(f.snapshot(), { tools: ["alpha"] });
        assert.deepEqual(names(explicit), ["alpha"]);
        assert.deepEqual(explicit.getAllTools().map(t => t.name), ["alpha"], "allowlist limits registration, not just initial activity");
        explicit.setActiveToolsByName(["alpha", "blocked", "read", "vanished"]);
        assert.deepEqual(names(explicit), ["alpha"]);
        let requests = 0;
        explicit.agent.streamFunction = (model, context) => {
            assert.deepEqual(getCurrentSystemMessage(context.messages)?.toolsAdded?.map(t => t.name), ["alpha"]);
            assert.ok(++requests <= 2);
            return requests === 1
                ? terminal(response(model, [{ type: "toolCall", id: "allowlist-denied", name: "blocked", arguments: {} }], "toolUse"))
                : terminal(response(model, [{ type: "text", text: "restriction retained" }]));
        };
        await explicit.prompt("A model naming a denied tool is not permission");
        const denied = explicit.messages.find(m => m.role === "toolResult" && m.toolCallId === "allowlist-denied");
        assert.ok(denied?.role === "toolResult" && denied.isError);
        assert.equal(requests, 2);
        for (const options of [{ tools: [] }, { noTools: "all" as const }]) {
            const empty = await f.create(f.snapshot(), options);
            assert.deepEqual(names(empty), []);
            empty.setActiveToolsByName(["alpha", "blocked", "read"]);
            assert.deepEqual(names(empty), []);
            assert.deepEqual(empty.getAllTools(), []);
        }
        const modified = await f.create(f.snapshot(), { tools: ["-blocked", "+read"] });
        assert.deepEqual(names(modified), ["alpha", "read"], "modifiers apply to transcript loadout, not fresh defaults");
        const fresh = await f.create(SessionManager.inMemory(f.cwd), {}, [f.tool("alpha")]);
        assert.notEqual(fresh.sessionId, f.savedId);
        assert.deepEqual(names(fresh), ["alpha", "read"], "fresh session must not inherit another session's transcript");
        const explicitFresh = await f.create(SessionManager.inMemory(f.cwd), { tools: ["read"] });
        assert.deepEqual(names(explicitFresh), ["read"], "explicit tools replace rather than restore saved selection");
        assert.deepEqual(f.calls, { alpha: 0, blocked: 0, vanished: 0 });
    } finally { f.close(); }
});

test("1.3 error and public abort settle resumed sessions without reviving denied tools", { timeout: 15000 }, async () => {
    const f = await fixture();
    try {
        const session = await f.create(f.snapshot(), { excludeTools: ["blocked"] });
        let errors = 0;
        session.agent.streamFunction = model => {
            errors++;
            return terminal({ ...response(model, [], "error"), errorMessage: "SYNTHETIC_FAILURE_NOT_PROVIDER_USAGE" });
        };
        await session.prompt("Synthetic failure");
        assert.equal(errors, 1, "fixture retry is disabled");
        assert.equal(session.messages.at(-1)?.role, "assistant");
        const failed = session.messages.at(-1);
        assert.ok(failed?.role === "assistant");
        assert.equal(failed.stopReason, "error");
        assert.equal(session.isIdle, true);
        assert.deepEqual(names(session), ["alpha"]);
        let entered: (() => void) | undefined;
        const started = new Promise<void>(resolve => { entered = resolve; });
        let abortObserved = 0;
        let invocations = 0;
        session.agent.streamFunction = (model, context, options) => {
            invocations++;
            assert.deepEqual(getCurrentSystemMessage(context.messages)?.toolsAdded?.map(t => t.name), ["alpha"]);
            const stream = createAssistantMessageEventStream();
            const signal = options?.signal;
            assert.ok(signal);
            const unavailable: Usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
            const abort = () => {
                abortObserved++;
                stream.push({ type: "error", reason: "aborted", error: {
                    ...response(model, [], "aborted"), usage: unavailable,
                    errorMessage: "Synthetic abort: zeros are unavailable-usage placeholders",
                } });
            };
            if (signal.aborted) abort();
            else signal.addEventListener("abort", abort, { once: true });
            entered?.();
            return stream;
        };
        const running = session.prompt("Hold synthetic stream until public abort");
        await started;
        assert.equal(session.isStreaming, true);
        await Promise.all([session.abort(), running]);
        assert.equal(invocations, 1);
        assert.equal(abortObserved, 1, "actual SDK AbortSignal must reach the synthetic stream");
        assert.equal(session.isIdle, true);
        assert.equal(session.pendingMessageCount, 0);
        const aborted = session.messages.at(-1);
        assert.ok(aborted?.role === "assistant");
        assert.equal(aborted.stopReason, "aborted");
        assert.deepEqual(names(session), ["alpha"]);
        assert.deepEqual(f.calls, { alpha: 0, blocked: 0, vanished: 0 });
        assert.ok(session.sessionFile);
        const reopened = await f.create(f.snapshot(session.sessionFile), { excludeTools: ["blocked"] });
        assert.deepEqual(names(reopened), ["alpha"], "failed/cancelled transcript cannot restore denied registration");
        reopened.setActiveToolsByName(["blocked", "vanished", "alpha"]);
        assert.deepEqual(names(reopened), ["alpha"]);
        let recoveryRequests = 0;
        reopened.agent.streamFunction = model => {
            assert.ok(++recoveryRequests <= 2);
            return recoveryRequests === 1
                ? terminal(response(model, [{ type: "toolCall", id: "after-abort", name: "alpha", arguments: {} }], "toolUse"))
                : terminal(response(model, [{ type: "text", text: "offline recovery" }]));
        };
        await reopened.prompt("Allowed custom registration still works after reopening");
        assert.equal(recoveryRequests, 2);
        assert.deepEqual(f.calls, { alpha: 1, blocked: 0, vanished: 0 });
        assert.equal(reopened.isIdle, true);
    } finally { f.close(); }
});

function sse(chunks: unknown[]): Response {
    return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n", {
        status: 200, headers: { "content-type": "text/event-stream" },
    });
}
async function catalogRuntime(cwd: string) {
    return ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
        modelsStorePath: join(cwd, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
}

test("1.3 Kimi installed catalog and real ChatCompletions route agree, with typed tools and usage", { timeout: 15000 }, async () => {
    pin130();
    const cwd = mkdtempSync(join(tmpdir(), "rosclaw-pi130-kimi-"));
    try {
        const runtime = await catalogRuntime(cwd);
        const data = json(join(aiRoot, "dist/providers/data/kimi-coding.json"));
        object(data);
        assert.deepEqual(Object.keys(data), ["openai-completions"]);
        const entries = data["openai-completions"];
        object(entries);
        const models = runtime.getModels("kimi-coding");
        assert.equal(models.length, Object.keys(entries).length);
        for (const entry of Object.values(entries)) {
            object(entry);
            assert.equal(typeof entry.id, "string");
            assert.ok(typeof entry.id === "string");
            const model = runtime.getModel("kimi-coding", entry.id);
            assert.ok(model);
            assert.equal(model.api, "openai-completions");
            assert.equal(model.baseUrl, "https://api.kimi.com/coding/v1");
            for (const key of ["id", "api", "provider", "baseUrl", "contextWindow", "maxTokens"] as const) {
                assert.deepEqual(model[key], entry[key], `installed catalog agreement: ${entry.id}/${key}`);
            }
        }
        const model = runtime.getModel("kimi-coding", "kimi-for-coding");
        assert.ok(model);
        assert.equal(model.api, "openai-completions");
        const context: Context = { messages: [{ role: "user", content: "Offline typed tool fixture", timestamp: 1 }],
            tools: [{ name: "inspect_fixture", description: "No side effects", parameters: Type.Object({ value: Type.Integer() }) }] };
        let fetches = 0;
        const fetch: typeof globalThis.fetch = async (input, init) => {
            fetches++;
            const request = new Request(input, init);
            assert.equal(request.url, "https://api.kimi.com/coding/v1/chat/completions", "not Anthropic messages or a /v1-less endpoint");
            assert.equal(request.method, "POST");
            const body: unknown = await request.json();
            object(body);
            assert.equal(body.model, "kimi-for-coding");
            assert.equal(body.stream, true);
            assert.ok(Array.isArray(body.messages));
            assert.ok(Array.isArray(body.tools));
            const declared = body.tools[0];
            object(declared);
            assert.equal(declared.type, "function");
            object(declared.function);
            assert.equal(declared.function.name, "inspect_fixture");
            object(declared.function.parameters);
            assert.equal(declared.function.parameters.type, "object");
            return sse([
                { id: "synthetic-kimi", object: "chat.completion.chunk", choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{ index: 0, id: "fixture-call", type: "function", function: { name: "inspect_fixture", arguments: '{"value":7}' } }] }, finish_reason: null }] },
                { id: "synthetic-kimi", object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
                { id: "synthetic-kimi", object: "chat.completion.chunk", choices: [], usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15, prompt_tokens_details: { cached_tokens: 3 } } },
            ]);
        };
        // No runtime key registration or auth-file access; this per-request synthetic
        // key and fetch substitute exercise the actual provider/API implementation.
        const stream = runtime.stream(model, context, { apiKey: "SYNTHETIC_OFFLINE", fetch, maxRetries: 0 });
        const events: string[] = [];
        for await (const event of stream) events.push(event.type);
        const result: AssistantMessage = await stream.result();
        assert.equal(fetches, 1);
        assert.ok(events.includes("toolcall_end"));
        assert.equal(events.at(-1), "done");
        assert.equal(result.api, "openai-completions");
        assert.equal(result.provider, "kimi-coding");
        assert.equal(result.stopReason, "toolUse");
        const call = result.content.find(block => block.type === "toolCall");
        assert.ok(call?.type === "toolCall");
        assert.equal(call.name, "inspect_fixture");
        assert.equal(call.id, "fixture-call");
        assert.deepEqual(call.arguments, { value: 7 });
        assert.deepEqual([result.usage.input, result.usage.cacheRead, result.usage.output, result.usage.totalTokens], [8, 3, 4, 15]);
        assert.equal(result.usage.cacheWrite, 0);
        assert.ok(Number.isFinite(result.usage.cost.total));
        assert.equal(typeof result.usage.cost.input, "number");
    } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("1.3 Kimi transport failure and pre-aborted request cannot become successful completions", { timeout: 15000 }, async () => {
    pin130();
    const cwd = mkdtempSync(join(tmpdir(), "rosclaw-pi130-kimi-negative-"));
    try {
        const runtime = await catalogRuntime(cwd);
        const model = runtime.getModel("kimi-coding", "kimi-for-coding");
        assert.ok(model);
        const context: Context = { messages: [{ role: "user", content: "negative offline fixture", timestamp: 1 }] };
        let fetches = 0;
        const failed = await runtime.complete(model, context, { apiKey: "SYNTHETIC_OFFLINE", maxRetries: 0,
            fetch: async () => {
                fetches++;
                return new Response(JSON.stringify({ error: { message: "SYNTHETIC_KIMI_FAILURE", type: "invalid_request_error" } }),
                    { status: 400, headers: { "content-type": "application/json" } });
            } });
        assert.equal(fetches, 1);
        assert.equal(failed.stopReason, "error");
        assert.match(failed.errorMessage ?? "", /SYNTHETIC_KIMI_FAILURE/);
        assert.deepEqual(failed.content, []);
        const controller = new AbortController();
        const reason = new DOMException("PI130_PREABORT_AUTH_CANCELLED", "AbortError");
        controller.abort(reason);
        assert.equal(controller.signal.aborted, true);
        assert.equal(controller.signal.reason, reason);
        // Probe the public auth boundary, not an arbitrary setup failure. Require
        // the supplied cancellation reason itself; unrelated errors must fail.
        await assert.rejects(runtime.getAuth(model, { apiKey: "SYNTHETIC_OFFLINE", signal: controller.signal }),
            (error: unknown) => { assert.equal(error, reason); return true; });
        let payloads = 0;
        const stream = runtime.stream(model, context, { apiKey: "SYNTHETIC_OFFLINE", maxRetries: 0,
            signal: controller.signal,
            onPayload: () => { payloads++; },
            fetch: async () => { fetches++; throw new Error("PREABORT_MUST_NOT_FETCH"); } });
        const events = [];
        for await (const event of stream) events.push(event);
        const cancelled = await stream.result();
        assert.equal(controller.signal.aborted, true);
        assert.equal(fetches, 1, "pre-aborted setup must not enter the synthetic fetch");
        assert.equal(payloads, 0, "auth cancellation precedes provider payload construction");
        // ModelRuntime awaits getAuth before provider.stream. lazyStream emits
        // error/error for setup rejection, not the provider catch's aborted.
        assert.deepEqual(events.map(event => event.type), ["error"]);
        const event = events[0];
        assert.ok(event?.type === "error");
        assert.equal(event.reason, "error");
        assert.strictEqual(event.error, cancelled);
        assert.equal(cancelled.stopReason, "error");
        assert.equal(cancelled.errorMessage, reason.message);
        assert.equal(cancelled.role, "assistant");
        assert.equal(cancelled.api, model.api);
        assert.equal(cancelled.provider, model.provider);
        assert.equal(cancelled.model, model.id);
        assert.deepEqual(cancelled.content, []);
        assert.deepEqual(cancelled.usage, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } });
        // Setup zeros are unavailable-usage placeholders, not recovered billing.
    } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("1.3 Codex gpt-6.1-sol catalog limits are metadata only", { timeout: 15000 }, async () => {
    pin130();
    const cwd = mkdtempSync(join(tmpdir(), "rosclaw-pi130-codex-metadata-"));
    try {
        const runtime = await catalogRuntime(cwd);
        const data = json(join(aiRoot, "dist/providers/data/openai-codex.json"));
        object(data);
        const group = data["openai-codex-responses"];
        object(group);
        const entry = group["chat:gpt-6.1-sol"];
        object(entry);
        const model = runtime.getModel("openai-codex", "gpt-6.1-sol");
        assert.ok(model);
        assert.equal(model.api, "openai-codex-responses");
        assert.equal(model.contextWindow, 500000);
        assert.equal(model.maxTokens, 128000);
        assert.equal(model.contextWindow, entry.contextWindow);
        assert.equal(model.maxTokens, entry.maxTokens);
        // Deliberately no stream, auth, compaction, cancellation or provider call.
    } finally { rmSync(cwd, { recursive: true, force: true }); }
});

// Unlike pre-aborted auth setup, this signal is live on entry to the actual
// provider onPayload boundary and is cancelled there before synthetic fetch.
test("1.3 Kimi provider-phase cancellation is aborted and the same runtime recovers", { timeout: 15000 }, async () => {
    pin130();
    const cwd = mkdtempSync(join(tmpdir(), "rosclaw-pi130-kimi-provider-abort-"));
    try {
        const runtime = await catalogRuntime(cwd);
        const model = runtime.getModel("kimi-coding", "kimi-for-coding");
        assert.ok(model);
        const context: Context = { messages: [{ role: "user", content: "provider cancellation fixture", timestamp: 1 }] };
        const controller = new AbortController();
        let payloads = 0;
        let fetches = 0;
        const stream = runtime.stream(model, context, { apiKey: "SYNTHETIC_OFFLINE", maxRetries: 0,
            signal: controller.signal,
            onPayload: (payload, actual) => {
                assert.equal(controller.signal.aborted, false, "setup must finish before cancellation");
                assert.equal(actual.api, "openai-completions");
                assert.equal(actual.provider, "kimi-coding");
                object(payload);
                assert.equal(payload.model, model.id);
                payloads++;
                controller.abort();
            },
            fetch: async () => { fetches++; throw new Error("PROVIDER_PREWIRE_ABORT_MUST_NOT_FETCH"); } });
        const events = [];
        for await (const event of stream) events.push(event);
        const cancelled = await stream.result();
        assert.equal(payloads, 1, "actual provider onPayload proves the setup/provider phase boundary");
        assert.equal(controller.signal.aborted, true);
        assert.equal(fetches, 0);
        assert.deepEqual(events.map(event => event.type), ["error"], "no start, delta or successful done");
        const event = events[0];
        assert.ok(event?.type === "error");
        assert.equal(event.reason, "aborted");
        assert.strictEqual(event.error, cancelled);
        assert.equal(cancelled.stopReason, "aborted");
        assert.match(cancelled.errorMessage ?? "", /abort/i);
        assert.deepEqual(cancelled.content, []);
        // This second request has a fresh live signal on the same runtime/model.
        // It must reach the real provider route and finish the synthetic SSE.
        const recoveryController = new AbortController();
        let recoveryPayloads = 0;
        const recovery = runtime.stream(model, context, { apiKey: "SYNTHETIC_OFFLINE", maxRetries: 0,
            signal: recoveryController.signal,
            onPayload: () => { recoveryPayloads++; assert.equal(recoveryController.signal.aborted, false); },
            fetch: async (input, init) => {
                fetches++;
                const request = new Request(input, init);
                assert.equal(request.url, "https://api.kimi.com/coding/v1/chat/completions");
                assert.equal(request.signal.aborted, false);
                return sse([
                    { id: "synthetic-recovery", object: "chat.completion.chunk", choices: [{ index: 0, delta: { role: "assistant", content: "offline recovery" }, finish_reason: null }] },
                    { id: "synthetic-recovery", object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
                ]);
            } });
        const recoveryEvents: string[] = [];
        for await (const event of recovery) recoveryEvents.push(event.type);
        const recovered = await recovery.result();
        assert.equal(recoveryPayloads, 1);
        assert.equal(fetches, 1, "only the fresh recovery request admits synthetic transport");
        assert.equal(recoveryEvents.includes("error"), false);
        assert.equal(recoveryEvents.at(-1), "done");
        assert.equal(recovered.stopReason, "stop");
        assert.equal(recovered.errorMessage, undefined);
        assert.deepEqual(recovered.content, [{ type: "text", text: "offline recovery" }]);
        assert.equal(controller.signal.aborted, true, "recovery does not reset or reuse the cancelled signal");
        // Neither cancellation zeros nor synthetic recovery usage are live billing.
    } finally { rmSync(cwd, { recursive: true, force: true }); }
});
