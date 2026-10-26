import assert from "node:assert/strict";
import test from "node:test";
import {OperationWatcher} from "../src/native/operation-watcher.js";

const pause = async () => {};

test("headless settlement consumes actual terminal events and same-session follow-up work", async () => {
	let polls = 0, sent = 0, waits = 0;
	const watcher = new OperationWatcher({
		call: async (method, params) => {
			if (method === "pi.op.get") return {operation: {task_id:"task",state:"RUNNING"}};
			if (method === "pi.kernel.get") return {task:{state:"RUNNING",active_revision:1}};
			if (method === "pi.kernel.events") {
				polls++;
				return {events: polls < 2 ? [] : [
					{seq:1,event_type:"operation.completed",operation_id:"op_first",payload:{state:"SUCCEEDED"}},
					...(polls > 2 ? [{seq:2,event_type:"operation.completed",operation_id:"op_follow",payload:{state:"SUCCEEDED"}}] : []),
				].filter(e => e.seq > Number(params.last_seq))};
			}
			return {};
		},
		sink: () => ({isIdle:true,api:{sendMessage: () => { sent++; }}}),
	});
	watcher.track("op_first");
	await watcher.settle({isStreaming:false,waitForIdle: async () => {
		waits++;
		if (sent === 1) watcher.track("op_follow");
	}}, pause);
	assert.equal(sent, 2);
	assert.ok(polls >= 3);
	assert.ok(waits >= 4, "two quiet rounds follow the last terminal");
	watcher.stop();
});

test("concurrent timer and settlement polls share one bridge request", async () => {
	let calls = 0;
	let release!: () => void;
	const pending = new Promise<void>(resolve => { release = resolve; });
	const watcher = new OperationWatcher({
		call: async method => {
			if (method === "pi.op.get") {
				calls++; await pending;
				return {operation:{task_id:"task",state:"SUCCEEDED"}};
			}
			if (method === "pi.kernel.get") return {task:{state:"SUCCEEDED"}};
			return {events:[]};
		}, sink: () => undefined,
	});
	watcher.track("op_first");
	const tick = watcher as unknown as {tick(): Promise<void>};
	const first = tick.tick(), second = tick.tick();
	assert.equal(first, second);
	assert.equal(calls, 1);
	release(); await Promise.all([first, second]); watcher.stop();
});

test("shutdown interrupts a stalled bridge without reporting completion", {timeout:1000}, async () => {
	const watcher = new OperationWatcher({call: () => new Promise(() => {}),sink:() => undefined});
	watcher.track("op_unknown");
	const settled = watcher.settle({isStreaming:false,waitForIdle:async () => {}},pause);
	watcher.stop();
	await assert.rejects(settled, /NATIVE_SESSION_SHUTDOWN/);
});

test("unavailable operation facts remain pending until shutdown", {timeout:1000}, async () => {
	let polls = 0;
	const watcher = new OperationWatcher({call:async () => {polls++;throw Error("bridge unavailable");},sink:() => undefined});
	watcher.track("op_unknown");
	await assert.rejects(watcher.settle({isStreaming:false,waitForIdle:async () => {}},async () => {
		if (polls === 3) watcher.stop();
	}), /NATIVE_SESSION_SHUTDOWN/);
	assert.equal(polls, 3);
});


test("settling an already closed watcher rejects without polling", async () => {
    let calls = 0;
    const watcher = new OperationWatcher({call:async () => {calls++;return {};},sink:() => undefined});
    watcher.stop();
    await assert.rejects(watcher.settle({isStreaming:false,waitForIdle:async () => {}},pause), /NATIVE_SESSION_SHUTDOWN/);
    assert.equal(calls, 0);
});

test("a late bridge reply after shutdown cannot trigger a model follow-up", async () => {
    let release!: (value: Record<string,unknown>) => void;
    let sent = 0;
    const response = new Promise<Record<string,unknown>>(resolve => {release = resolve;});
    const watcher = new OperationWatcher({call:() => response,
        sink:() => ({isIdle:true,api:{sendMessage:() => {sent++;}}})});
    watcher.track("op_pending");
    const pending = (watcher as unknown as {tick():Promise<void>}).tick();
    watcher.stop();
    release({operation:{task_id:"task",state:"SUCCEEDED"}});
    await pending;
    assert.equal(sent, 0);
});
