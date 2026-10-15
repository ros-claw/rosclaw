import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream, InMemoryCredentialStore, type Model, type AssistantMessage } from "@earendil-works/pi-ai";
import { createToolCallBudgetExtension } from "../src/harness/pi/tool-call-budget.js";

// One real SDK session; only the model stream is synthetic. No provider transport.
test("1.2 user/custom startup is busy, abort is deferred, notices do not accumulate", { timeout: 10000 }, async () => {
    const cwd = mkdtempSync(join(tmpdir(), "rosclaw-pi120-start-"));
    const starts: string[] = [];
    const contexts: string[] = [];
    const sentinel = "PI120_START_SENTINEL\nOwned intent: preserve offline permissions and empty tool loadout.";
    const permissions = "PI120_PERMISSIONS: synthetic offline only; no executable tools";
    let release: (() => void) | undefined;
    let entered: (() => void) | undefined;
    let gate: Promise<void> | undefined;
    let dispatches = 0;
    let streamInvocations = 0;
    let abortedInvocations = 0;
    let abortedSignal = false;
    let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
    const observer: ExtensionFactory = pi => {
        pi.on("before_agent_start", async event => {
            starts.push(event.messageType);
            assert.ok(session);
            assert.equal(session.isStreaming, true, "SDK must own startup busy state");
            if (event.messageType === "user") assert.equal(event.prompt, "user start");
            else assert.ok(event.message.customType.startsWith("pi120."));
            entered?.();
            if (gate) await gate;
            return { systemPrompt: `${event.systemPrompt}\n${sentinel}\n${permissions}` };
        });
    };
    const budget = createToolCallBudgetExtension({ allowedTools: [], visibleBudget: true,
        modelUsageAwareness: { version: 1, lifetime: "runtime", inclusiveInputLimit: 10000,
            outputLimit: 1000, deliveryReserveInput: 10, deliveryReserveOutput: 10 } });
    const model: Model<"openai-completions"> = { id: "startup", name: "Offline startup", api: "openai-completions", provider: "startup-fixture",
        baseUrl: "http://invalid.local", reasoning: false, input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 100 };
    try {
        const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
        const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null,
            modelsStorePath: join(cwd, "catalog"), allowModelNetwork: false, refreshOnCreate: false });
        runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, apiKey: "SYNTHETIC_OFFLINE", models: [model] });
        const selected = runtime.getModel(model.provider, model.id);
        assert.ok(selected);
        const loader = new DefaultResourceLoader({ cwd, agentDir: cwd, settingsManager: settings,
            extensionFactories: [{ name: "budget", factory: budget }, { name: "observe", factory: observer }],
            noExtensions: true, noContextFiles: true, noSkills: true, noPromptTemplates: true, noThemes: true });
        await loader.reload();
        assert.deepEqual(loader.getExtensions().errors, []);
        ({ session } = await createAgentSession({ cwd, agentDir: cwd, resourceLoader: loader,
            modelRuntime: runtime, settingsManager: settings, sessionManager: SessionManager.inMemory(cwd),
            model: selected, tools: [], thinkingLevel: "off" }));
        await session.bindExtensions({});
        session.agent.streamFunction = (actualModel, context, options) => {
            ++streamInvocations;
            // Invocation is not admission: inspect the SDK's real signal before
            // the synthetic transport boundary or any successful usage is emitted.
            if (options?.signal?.aborted) {
                ++abortedInvocations;
                abortedSignal = options.signal.aborted;
                const stream = createAssistantMessageEventStream();
                const message: AssistantMessage = { role: "assistant", content: [],
                    api: actualModel.api, provider: actualModel.provider, model: actualModel.id,
                    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                    stopReason: "aborted", errorMessage: "Synthetic request aborted before admission; usage unavailable",
                    timestamp: Date.now() };
                // Setup failure may terminate directly with error, without start/done.
                stream.push({ type: "error", reason: "aborted", error: message });
                return stream;
            }
            assert.ok(++dispatches <= 3, "no unintended model turn");
            assert.equal(actualModel.id, selected.id);
            assert.equal(actualModel.provider, selected.provider);
            assert.ok(options?.signal);
            const systems = context.messages.filter(message => message.role === "system");
            assert.equal(systems.length, 1, "final request has one leading system");
            assert.equal(context.messages[0]?.role, "system");
            const head = systems[0]!;
            assert.deepEqual((head.toolsAdded ?? []).map(tool => tool.name), [], "empty executable declaration fixture");
            const text = typeof head.content === "string" ? head.content : head.content.map(part => part.text).join("\n");
            contexts.push(text);
            assert.ok(text.includes(sentinel), "complete owned startup intent survives forced projection");
            assert.ok(text.includes(permissions), "owned permissions survive final projection");
            if (dispatches === 1) {
                assert.ok(session);
                session.agent.steer({ role: "user", content: [{ type: "text", text: "subsequent request" }], timestamp: Date.now() });
            }
            const stream = createAssistantMessageEventStream();
            const message: AssistantMessage = { role: "assistant", content: [{ type: "text", text: "synthetic" }],
                api: actualModel.api, provider: actualModel.provider, model: actualModel.id,
                usage: { input: 2, output: 1, cacheRead: 3, cacheWrite: 4, totalTokens: 10,
                    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                stopReason: "stop", timestamp: Date.now() };
            stream.push({ type: "start", partial: message });
            stream.push({ type: "done", reason: "stop", message });
            return stream;
        };
        assert.equal(session.isStreaming, false);
        await session.sendCustomMessage({ customType: "pi120.no-turn", content: "context only", display: false }, { triggerTurn: false });
        assert.equal(dispatches, 0, "non-triggering custom messages do not dispatch or enter startup");
        assert.deepEqual(starts, []);
        await session.prompt("user start");
        assert.equal(session.isStreaming, false, "user startup busy state is released at settlement");
        assert.deepEqual(starts, ["user"]);
        assert.equal(dispatches, 2, "genuine steering produces a subsequent request without another startup");
        await session.sendCustomMessage({ customType: "pi120.test", content: "custom start", display: false }, { triggerTurn: true });
        assert.deepEqual(starts, ["user", "custom"]);
        assert.equal(dispatches, 3);
        assert.equal(contexts.length, 3, "capture every actual stream request");
        const marker = "ROSCLAW_MODEL_USAGE_JSON:";
        for (const context of contexts) {
            assert.equal(context.split(marker).length - 1, 1);
            assert.equal(context.split("ROSCLAW_TOOL_POLICY_JSON:").length - 1, 1);
            assert.equal(context.split("PI120_START_SENTINEL").length - 1, 1);
        }
        for (const [index, context] of contexts.entries()) {
            const usageLine = context.split("\n").find(line => line.startsWith(marker));
            assert.ok(usageLine);
            const usage: unknown = JSON.parse(usageLine.slice(marker.length));
            assert.ok(typeof usage === "object" && usage !== null && "knownInclusiveInput" in usage && "completedMainRequests" in usage);
            assert.equal(usage.knownInclusiveInput, index * 9, "fresh final-request cache input counted once");
            assert.equal(usage.completedMainRequests, index, "genuine settled requests, not startup snapshots");
            const policyLine = context.split("\n").find(line => line.startsWith("ROSCLAW_TOOL_POLICY_JSON:"));
            assert.ok(policyLine);
            const policy: unknown = JSON.parse(policyLine.slice("ROSCLAW_TOOL_POLICY_JSON:".length));
            assert.ok(typeof policy === "object" && policy !== null && "allowedTools" in policy && "usedTotal" in policy);
            assert.deepEqual(policy.allowedTools, []);
            assert.equal(policy.usedTotal, 0);
        }
        const startupEntered = new Promise<void>(resolve => { entered = resolve; });
        gate = new Promise<void>(resolve => { release = resolve; });
        const turning = session.sendCustomMessage({ customType: "pi120.abort", content: "abort startup", display: false }, { triggerTurn: true });
        await startupEntered;
        assert.equal(session.isStreaming, true);
        const aborting = session.abort();
        release!();
        await Promise.all([turning, aborting]);
        assert.equal(dispatches, 3, "deferred abort prevents synthetic transport dispatch");
        assert.equal(session.isStreaming, false);
        assert.equal(session.pendingMessageCount, 0);
        assert.equal(starts.length, 3, "startup hook runs once, never enqueues an extra turn");
        assert.equal(streamInvocations, 4, "retain the real fourth SDK callback, distinct from transport admission");
        assert.equal(abortedInvocations, 1, "exactly one pre-aborted callback has zero synthetic admission");
        assert.equal(abortedSignal, true, "the fourth callback observes the actual SDK options.signal.aborted");
        assert.equal(streamInvocations - abortedInvocations, dispatches, "only non-aborted callbacks cross synthetic admission");
        assert.equal(session.agent.state.messages.at(-1)?.role, "assistant", "cancelled response settles into the actual SDK transcript");
        assert.equal(session.agent.state.messages.map(message => message.role === "assistant" ? message.stopReason : undefined).at(-1), "aborted", "cancelled settlement is not a successful fake response");
    } finally {
        release?.();
        session?.dispose();
        rmSync(cwd, { recursive: true, force: true });
    }
});
