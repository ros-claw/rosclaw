/** Real extension event wiring: watchdog abort, timeout card and user-cancel boundary. */
import assert from "node:assert/strict";
import {mkdtempSync, rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";
type Handler = (event: unknown, ctx: unknown) => Promise<unknown>;

async function buildExtension(
	home: string,
	probe?: () => { isolationReady: boolean },
) {
	const { createRosclawExtension } = await import("../src/extension/index.js");
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { AgentSessionCoordinator } = await import("../src/session/coordinator.js");
	const { SessionLeaseManager } = await import("../src/session/lease-manager.js");
	const { ProductStateCenter } = await import("../src/session/state-center.js");
	const { LocaleManager } = await import("../src/i18n/locale.js");
	const { resolveTaskContext } = await import("../src/native/active-task-context.js");

	const handlers = new Map<string, Handler[]>();
	const entries: Array<{type:string; data:unknown}> = [];
	const calls: string[] = [];
	const pi = {
		on(name: string, handler: Handler) {
			const list = handlers.get(name) ?? [];
			list.push(handler);
			handlers.set(name, list);
		},
		registerCommand() {},
		registerShortcut() {},
		registerEntryRenderer() {},
		registerMessageRenderer() {},
		appendEntry(type:string, data:unknown) { entries.push({type, data}); },
	};
	const active = new ActiveSessionContext({
		sessionId: "pi_test", missionId: undefined, contextRevision: 0,
		mode: "SIMULATION", profile: "developer", contextState: "LOADING",
		leaseState: "NONE", actionsAllowed: false,
	});
	const call = async (_home:string, method:string) => {
		calls.push(method);
		return method === "pi.kernel.active" ? {ok:true, task:{task_id:"task-1"}} : {ok:false};
	};
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: home, active,
		leaseManager: new SessionLeaseManager(home, call),
		notify: () => undefined, call,
	});
	const center = new ProductStateCenter({
		rosclawHome: home, active,
		operatorSocket: join(home, "run", "operatord.sock"),
		productVersion: "0.1.0", call: call as never,
		operatorCallFn: async () => ({ ok: false }),
	});
	const locale = new LocaleManager(join(home, "agent"));
	const factory = createRosclawExtension({
		profile: "developer", version: "0.1.0", systemPrompt: "TEST",
		active, coordinator, center, locale, rosclawHome: home,
		taskContext: resolveTaskContext({ rosclawHome: home, cwd: "/tmp", mode: "SIMULATION" }),
		...(probe ? { osIsolationProbe: probe } : {}),
	});
	factory(pi as never);
	return {handlers, entries, calls, center};
}


for (const errorMessage of [undefined, "Request was aborted"]) {
	test(`watchdog message_end retains timeout diagnosis (errorMessage=${errorMessage})`, async (t) => {
		const home = mkdtempSync(join(tmpdir(), "watchdog-reason-"));
		const {handlers, entries, calls, center} = await buildExtension(home);
		const notices: string[] = [];
		let aborts = 0;
		const ctx = {hasUI:false, isIdle:() => false, abort:() => {aborts++;}, ui:{notify:(text:string) => notices.push(text)}};
		const emit = async (event:string, payload:unknown = {}) => {
			for (const h of handlers.get(event) ?? []) await h(payload, ctx);
		};
		t.mock.timers.enable({apis:["setTimeout"]});
		try {
			await emit("session_start");
			await emit("turn_start");
			t.mock.timers.tick(30_001);
			assert.equal(aborts, 1);
			await emit("message_end", {message:{role:"assistant", stopReason:"aborted", errorMessage}});
			assert.ok(notices.some(n => n.includes("PROVIDER_RESPONSE_TIMEOUT")));
			assert.ok(!notices.some(n => n.includes("MODEL_UNKNOWN")));
			assert.ok(!calls.includes("pi.session.interrupt"), "provider stall must not become a user motion/operation interrupt");
			const error = entries.find(e => e.type === "rosclaw.provider_error");
			assert.equal((error?.data as {code:string}).code, "PROVIDER_RESPONSE_TIMEOUT");
			assert.equal(center.snapshot().provider, "PAUSED");
			// Next user abort is not inherited as another watchdog timeout.
			await emit("turn_start");
			await emit("message_end", {message:{role:"assistant", stopReason:"aborted"}});
			assert.ok(calls.includes("pi.session.interrupt"));
			await emit("turn_end", {message:{role:"assistant", content:[]}});
		} finally {
			await emit("session_shutdown");
			t.mock.timers.reset();
			rmSync(home, {recursive:true, force:true});
		}
	});
}
