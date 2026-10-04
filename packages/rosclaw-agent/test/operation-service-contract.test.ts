import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createAssistantMessageEventStream, InMemoryCredentialStore, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { OperationWatcher } from "../src/native/operation-watcher.js";

test("public SDK: service READY stdout stays observational; finite terminal triggers a real new agent turn", { timeout: 10_000 }, async () => {
    const home = await mkdtemp(join(tmpdir(), "rosclaw-service-contract-"));
    const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
    const history = SessionManager.inMemory(home);
    const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
        modelsStorePath: join(home, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
    const model: Model<"openai-completions"> = { id: "fixture", name: "fixture", api: "openai-completions",
        provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 100 };
    runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl,
        apiKey: "PRIVATE_FIXTURE_NOT_A_REAL_KEY", models: [model] });
    const loader = new DefaultResourceLoader({ cwd: home, agentDir: home, settingsManager: settings,
        noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true });
    await loader.reload();
    const { session } = await createAgentSession({ cwd: home, agentDir: home,
        model: runtime.getModel(model.provider, model.id)!, modelRuntime: runtime, sessionManager: history,
        settingsManager: settings, resourceLoader: loader, noTools: "all", thinkingLevel: "off" });
    let requests = 0;
    const lifecycle: string[] = [];
    session.subscribe(event => lifecycle.push(event.type));
    session.agent.streamFunction = () => {
        assert.ok(++requests <= 2, "notification caused an unbounded agent loop");
        const stream = createAssistantMessageEventStream();
        stream.push({ type: "done", reason: "stop", message: { role: "assistant", api: model.api,
            provider: model.provider, model: model.id, content: [{ type: "text", text: "Private offline turn complete" }],
            usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
            stopReason: "stop", timestamp: Date.now() } });
        return stream;
    };
    let state = "RUNNING", seq = 0, eventConsumed = false, sends = 0;
    const publications: Promise<void>[] = [];
    const widgets: unknown[] = [];
    const watcher = new OperationWatcher({
        call: async method => {
            if (method === "pi.op.get") return { operation: { operation_id: "op_fixture",
                task_id: "task_fixture", revision: 1, state } };
            if (method === "pi.kernel.get") return { task: { task_id: "task_fixture", state: "RUNNING", active_revision: 1 } };
            if (method === "pi.kernel.events") {
                if (eventConsumed) return { events: [] };
                eventConsumed = true;
                return { events: [{ seq: ++seq, operation_id: "op_fixture",
                    event_type: state === "RUNNING" ? "operation.output" : "operation.completed",
                    payload: state === "RUNNING" ? { text: "READY private fixture" } : { state } }] };
            }
            throw Error(method);
        },
        sink: () => ({ isIdle: session.isIdle, api: { sendMessage: (message, options) => {
            sends++;
            publications.push(session.sendCustomMessage(message, options));
        } }, setWidget: (key, lines) => widgets.push({ key, lines }) }),
    });
    const tick = () => (watcher as unknown as { tick(): Promise<void> }).tick();
    try {
        await session.prompt("Private offline initial turn; no real provider request.");
        assert.equal(requests, 1);
        watcher.track("op_fixture");
        await tick(); await tick();
        assert.equal(requests, 1, "stdout readiness-looking text improperly started a turn");
        assert.equal(sends, 0);
        assert.ok(widgets.length > 0, "service output was not projected to the widget");
        assert.ok(session.isIdle);
        state = "SUCCEEDED"; eventConsumed = false;
        await tick(); await Promise.all(publications); await tick();
        assert.equal(requests, 2, "terminal notification was merely a toast rather than a new SDK turn");
        assert.equal(sends, 1);
        assert.equal(lifecycle.filter(event => event === "agent_start").length, 2);
        assert.equal(lifecycle.filter(event => event === "agent_end").length, 2);
        const notifications = history.getEntries().filter(entry => entry.type === "custom_message");
        assert.equal(notifications.length, 1);
        assert.equal(notifications[0].customType, "rosclaw.operation.result");
        assert.ok(session.isIdle);
    } finally {
        session.dispose();
        await rm(home, { recursive: true, force: true });
    }
});
