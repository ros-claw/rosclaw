import assert from "node:assert/strict";
import test from "node:test";
import { OperationWatcher } from "../src/native/operation-watcher.js";

function fixture() {
    const state = { idle: false, operation: "RUNNING", task: "RUNNING", revision: 1,
        sendFailures: 0, kernelFailures: 0, kernelHook: undefined as (() => Promise<void>) | undefined };
    const sent: unknown[] = [];
    let seq = 0;
    const watcher = new OperationWatcher({
        call: async (method) => {
            if (method === "pi.op.get") return {operation: {
                operation_id: "op_aaa", task_id: "task_1", revision: 1, state: state.operation,
            }};
            if (method === "pi.kernel.get") {
                if (state.kernelFailures-- > 0) throw Error("bridge temporarily unavailable");
                await state.kernelHook?.();
                return {task: {
                task_id: "task_1", state: state.task, active_revision: state.revision,
            }};
            }
            if (method === "pi.kernel.events") {
                if (state.operation === "SUCCEEDED" && seq === 0) {
                    seq++;
                    return {events: [{seq, operation_id: "op_aaa", event_type: "operation.completed",
                        payload: {state: "SUCCEEDED"}}]};
                }
                return {events: []};
            }
            throw Error(method);
        },
        sink: () => ({isIdle: state.idle, api: {sendMessage: (message) => {
            if (state.sendFailures-- > 0) throw Error("session turn cancelled during publication");
            sent.push(message); }}}),
    });
    const tick = () => (watcher as unknown as {tick(): Promise<void>}).tick();
    const ack = (overrides: Record<string, unknown> = {}) => {
        (watcher as unknown as {observeToolResult(result: Record<string, unknown>): void}).observeToolResult({
            ok: true, status: "SUCCEEDED", operation: {
                operation_id: "op_aaa", task_id: "task_1", revision: 1, state: "SUCCEEDED",
            }, ...overrides,
        });
    };
    const completeWhileBusy = async () => {
        watcher.track("op_aaa"); await tick();
        state.operation = "SUCCEEDED"; await tick();
    };
    return {watcher, state, sent, tick, ack, completeWhileBusy};
}

test("busy terminal stays watcher-owned; explicit terminal consumption prevents a duplicate turn", async () => {
    const f = fixture(); await f.completeWhileBusy();
    assert.equal(f.sent.length, 0, "busy operation was irreversibly queued in PI");
    f.ack(); f.state.idle = true; await f.tick();
    assert.equal(f.sent.length, 0);
    f.watcher.track("op_aaa"); await f.tick();
    assert.equal(f.sent.length, 0, "consumed operation was rearmed");
});

test("unconsumed terminal survives stop/start and reaches idle exactly once", async () => {
    const f = fixture(); await f.completeWhileBusy();
    assert.equal(f.sent.length, 0);
    f.watcher.stop(); f.watcher.start(); f.watcher.stop();
    f.state.idle = true; await f.tick(); await f.tick();
    assert.equal(f.sent.length, 1);
});

for (const [label, result] of Object.entries({
    running: {status: "RUNNING", operation: {operation_id: "op_aaa", task_id: "task_1", revision: 1, state: "RUNNING"}},
    failed_read: {ok: false},
    wrong_operation: {operation: {operation_id: "op_other", task_id: "task_1", revision: 1, state: "SUCCEEDED"}},
    wrong_task: {operation: {operation_id: "op_aaa", task_id: "task_other", revision: 1, state: "SUCCEEDED"}},
    wrong_revision: {operation: {operation_id: "op_aaa", task_id: "task_1", revision: 2, state: "SUCCEEDED"}},
})) {
    test(`${label} cannot consume the pending terminal reminder`, async () => {
        const f = fixture(); await f.completeWhileBusy(); f.ack(result);
        f.state.idle = true; await f.tick(); assert.equal(f.sent.length, 1);
    });
}

test("revision change while busy archives the old operation without a turn", async () => {
    const f = fixture(); await f.completeWhileBusy(); f.state.revision = 2;
    f.state.idle = true; await f.tick(); assert.equal(f.sent.length, 0);
});

test("structured bridge status/identity preserves other metadata and consumes a terminal read", async () => {
    const {executeVia} = await import("../src/tools/bridge-tools.js");
    const f = fixture(); await f.completeWhileBusy();
    const operation = {operation_id: "op_aaa", task_id: "task_1", revision: 1, state: "SUCCEEDED"};
    const result = await executeVia({
        active: {current: {missionId: "m1", sessionId: "s1", contextRevision: 1, mode: "SIMULATION"}},
        center: {call: async () => ({ok: true, result: {
            ok: true, status: "SUCCEEDED", summary: "done", operation,
            evidence_refs: ["evidence_1"], artifact_refs: ["artifact_1"],
        }})},
    } as never, "rosclaw_process_output", {operation_id: "op_aaa"});
    assert.deepEqual(result.details.operation, operation);
    assert.deepEqual(result.details.evidence_refs, ["evidence_1"]);
    assert.deepEqual(result.details.artifact_refs, ["artifact_1"]);
    f.watcher.observeToolResult(result.details);
    f.state.idle = true; await f.tick(); assert.equal(f.sent.length, 0);
});

test("a terminal read before watcher registration resolves still consumes only the matching operation", async () => {
    const f = fixture(); f.watcher.track("op_aaa");
    f.ack(); f.state.operation = "SUCCEEDED"; f.state.idle = true; await f.tick();
    assert.equal(f.sent.length, 0);
});

for (const failure of ["sendFailures", "kernelFailures"] as const) {
    test(`${failure}: interrupted publication or temporary bridge failure retains the reminder`, async () => {
        const f = fixture(); await f.completeWhileBusy(); f.state.idle = true;
        f.state[failure] = 1; await f.tick(); assert.equal(f.sent.length, 0);
        await f.tick(); assert.equal(f.sent.length, 1);
    });
}

test("terminal consumption during ownership await cannot race into a duplicate notification", async () => {
    const f = fixture(); await f.completeWhileBusy(); f.state.idle = true;
    f.state.kernelHook = async () => { f.ack(); };
    await f.tick(); assert.equal(f.sent.length, 0);
});

for (const envelope of [{ok: true, innerOk: false}, {ok: false, innerOk: true}]) {
    test(`failed bridge envelope ${JSON.stringify(envelope)} cannot consume a reminder`, async () => {
        const {executeVia} = await import("../src/tools/bridge-tools.js");
        const f = fixture(); await f.completeWhileBusy();
        const result = await executeVia({
            active: {current: {missionId: "m1", sessionId: "s1", contextRevision: 1, mode: "SIMULATION"}},
            center: {call: async () => ({ok: envelope.ok, result: {
                ok: envelope.innerOk, status: "SUCCEEDED", summary: "fixture error",
                operation: {operation_id: "op_aaa", task_id: "task_1", revision: 1, state: "SUCCEEDED"},
            }})},
        } as never, "rosclaw_process_output", {operation_id: "op_aaa"});
        assert.equal(result.details.ok, false); assert.equal(result.isError, true);
        f.watcher.observeToolResult(result.details);
        f.state.idle = true; await f.tick(); assert.equal(f.sent.length, 1);
    });
}

for (const successfulRead of [true, false]) {
    test(`real public PI abort during ${successfulRead ? "completed" : "failed"} terminal read preserves the right reminder`, async () => {
        const {Agent} = await import("@earendil-works/pi-agent-core");
        const {createAssistantMessageEventStream, Type} = await import("@earendil-works/pi-ai");
        const f = fixture(); await f.completeWhileBusy();
        const model = {id: "fixture", name: "fixture", api: "openai-completions" as const,
            provider: "fixture", baseUrl: "http://invalid.local", reasoning: false, input: ["text" as const],
            cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}, contextWindow: 4096, maxTokens: 100};
        let requests = 0;
        const agent = new Agent({
            initialState: {model, tools: [{name: "process_output", label: "Fixture", description: "private read",
                parameters: Type.Object({operation_id: Type.String()}),
                execute: async () => {
                    agent.abort();
                    if (!successfulRead) throw Error("fixture read aborted before result");
                    return {content: [{type: "text" as const, text: "fixture output"}], details: {
                        ok: true, status: "SUCCEEDED", operation: {
                            operation_id: "op_aaa", task_id: "task_1", revision: 1, state: "SUCCEEDED",
                        },
                    }};
                }}]},
            streamFn: (_model, _context, options) => {
                if (++requests > 2) throw Error("fixture exceeded bounded model requests");
                const stream = createAssistantMessageEventStream();
                const message = {role: "assistant" as const, api: model.api, provider: model.provider, model: model.id,
                    content: [{type: "toolCall" as const, id: "fixture_tool_call", name: "process_output", arguments: {operation_id: "op_aaa"}}],
                    usage: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0}},
                    stopReason: "toolUse" as const, timestamp: 1};
                if (options?.signal?.aborted) {
                    stream.push({type: "error", reason: "aborted", error: {
                        ...message, content: [], stopReason: "aborted", errorMessage: "fixture aborted",
                    }});
                } else {
                    stream.push({type: "done", reason: "toolUse", message});
                }
                return stream;
            },
        });
        agent.subscribe(event => {
            if (event.type === "message_end") f.watcher.observeToolMessage(event.message as unknown as Record<string, unknown>);
        });
        await agent.prompt("read private fixture output");
        const saved = agent.state.messages.find(message => message.role === "toolResult");
        assert.ok(saved, "cancelled PI run failed to retain the finalized tool result");
        assert.equal((saved as {isError?: boolean}).isError, !successfulRead, JSON.stringify(saved));
        f.state.idle = true; await f.tick();
        assert.equal(f.sent.length, successfulRead ? 0 : 1);
    });
}

test("unrelated message metadata cannot acknowledge an operation result", async () => {
    const f = fixture(); await f.completeWhileBusy();
    const details = {ok: true, status: "SUCCEEDED", operation: {
        operation_id: "op_aaa", task_id: "task_1", revision: 1, state: "SUCCEEDED",
    }};
    for (const message of [
        {role: "user", toolName: "process_output", details},
        {role: "toolResult", toolName: "read", details},
        {role: "toolResult", toolName: "process_output", details, isError: true},
    ]) f.watcher.observeToolMessage(message);
    f.state.idle = true; await f.tick(); assert.equal(f.sent.length, 1);
});
