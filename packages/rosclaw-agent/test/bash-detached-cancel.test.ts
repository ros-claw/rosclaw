import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { BashProcessOwnership, readProcessBirth, sameProcessBirth, signalBashFallback, type ProcessBirth } from "../src/tools/bash-process-ownership.js";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
function identity(pid: number) {
	try {
		const text = readFileSync(`/proc/${pid}/stat`, "utf8");
		const fields = text.slice(text.lastIndexOf(")") + 2).split(" ");
		return { start: fields[19], running: fields[0] !== "Z" && fields[0] !== "X" };
	} catch { return undefined; }
}
function cleanup(pid: number, start: string) {
	const current = identity(pid);
	if (current?.start === start && current.running) process.kill(pid, "SIGKILL");
}
async function until(check: () => boolean) {
	for (let i = 0; i < 300; i++) { if (check()) return; await delay(10); }
	assert.fail("private fixture did not become ready");
}

for (const kind of ["abort", "timeout"] as const) {
	test(`bash ${kind} drains a detached descendant without killing unrelated work`, { skip: process.platform !== "linux", timeout: 8000 }, async () => {
		const root = mkdtempSync(join(tmpdir(), "rosclaw-detached-cancel-"));
		const marker = join(root, "child.pid");
		const childCode = `process.on('SIGTERM',()=>{});require('fs').writeFileSync(${JSON.stringify(marker)}, String(process.pid)); setInterval(() => {},1000)`;
		const script = join(root, "parent.cjs");
		writeFileSync(script, `require('child_process').spawn(process.execPath,['-e',${JSON.stringify(childCode)}],{detached:true,stdio:'ignore'}).unref();setInterval(()=>{},1000)`);
		const unrelated = spawn(process.execPath, ["-e", "setInterval(()=>{},1000)"], { stdio: "ignore" });
		assert.ok(unrelated.pid);
		const unrelatedBirth = identity(unrelated.pid)!;
		const tool = buildWorkspacePackTools({ root, bwrapPath: () => null }).find((t) => t.name === "bash")!;
		const control = new AbortController();
		let pid: number | undefined, start: string | undefined;
		try {
			const pending = tool.execute("detached", { command: `${JSON.stringify(process.execPath)} ${JSON.stringify(script)}`,
				...(kind === "timeout" ? { timeout_sec: 0.7 } : {}) }, control.signal, undefined, {} as never);
			await until(() => existsSync(marker));
			pid = Number(readFileSync(marker, "utf8")); start = identity(pid)!.start;
			if (kind === "abort") control.abort();
			const result = await pending;
			assert.equal(result.isError, true);
			assert.match(JSON.stringify(result.content), kind === "abort" ? /ABORTED/ : /TIMEOUT/);
			assert.equal((result.details as { processCleanup?: { status: string } }).processCleanup?.status, "captured_descendants_stopped");
			assert.ok(!identity(pid)?.running, "cancel returned with a live detached descendant");
			assert.deepEqual(identity(unrelated.pid), unrelatedBirth, "unrelated process must remain alive");
		} finally {
			control.abort();
			if (pid && start) cleanup(pid, start);
			cleanup(unrelated.pid, unrelatedBirth.start);
			await new Promise<void>((resolve) => unrelated.once("close", () => resolve()));
		}
	});
}

test("normal bash completion preserves deliberately detached work", { skip: process.platform !== "linux", timeout: 5000 }, async () => {
	const root = mkdtempSync(join(tmpdir(), "rosclaw-normal-detach-"));
	const marker = join(root, "child.pid");
	const script = join(root, "parent.cjs");
	writeFileSync(script, `require('child_process').spawn(process.execPath,['-e',${JSON.stringify(`require('fs').writeFileSync(${JSON.stringify(marker)},String(process.pid));setInterval(()=>{},1000)`)}],{detached:true,stdio:'ignore'}).unref()`);
	const tool = buildWorkspacePackTools({ root, bwrapPath: () => null }).find((t) => t.name === "bash")!;
	let pid: number | undefined, start: string | undefined;
	try {
		const result = await tool.execute("normal-detach", { command: `${JSON.stringify(process.execPath)} ${JSON.stringify(script)}` }, undefined, undefined, {} as never);
		await until(() => existsSync(marker));
		pid = Number(readFileSync(marker, "utf8")); start = identity(pid)!.start;
		assert.equal(result.isError, false);
		assert.ok(identity(pid)?.running);
	} finally { if (pid && start) cleanup(pid, start); }
});

test("birth comparisons reject PID reuse and changed UID or boot", () => {
	const a: ProcessBirth = { pid: 100, ppid: 99, start: "200", uid: 1000, boot: "a", state: "S" };
	assert.equal(sameProcessBirth(a, { ...a, ppid: 1 }), true);
	for (const b of [{ ...a, pid: 101 }, { ...a, start: "201" }, { ...a, uid: 1001 }, { ...a, boot: "b" }]) {
		assert.equal(sameProcessBirth(a, b), false);
	}
});

test("missing process identity grants no cleanup success", { skip: process.platform !== "linux" }, async () => {
	const owner = new BashProcessOwnership(2147483647);
	assert.equal((await owner.stop(true)).status, "unresolved");
});

test("non-Linux Unix still signals the shell process group; Windows signals the child", () => {
	const sent: unknown[] = [];
	const child = { pid: 1234, kill: (sig?: NodeJS.Signals | number) => { sent.push(["child", sig]); return true; } };
	signalBashFallback(child, "SIGTERM", "darwin", (pid, sig) => sent.push([pid, sig]));
	signalBashFallback(child, "SIGKILL", "win32", (pid, sig) => sent.push([pid, sig]));
	assert.deepEqual(sent, [[-1234, "SIGTERM"], ["child", "SIGKILL"]]);
});

test("identity read failures distinguish absent processes from unknown state", () => {
	for (const code of ["ENOENT", "ESRCH", "EACCES", "EIO"]) {
		const result = readProcessBirth(1234, "boot", () => { throw Object.assign(new Error("fixture"), { code }); });
		assert.equal(result.kind, code === "ENOENT" || code === "ESRCH" ? "gone" : "unresolved");
	}
	assert.equal(readProcessBirth(1234, "boot", () => "bad stat").kind, "unresolved");
});

test("permission failure after capture is unresolved and cancellation is singleflight", { skip: process.platform !== "linux" }, async () => {
	let calls = 0;
	const owner = new BashProcessOwnership(2147483647, (pid, boot) => {
		if (!calls++) return { kind: "observed", birth: { pid, boot, ppid: 1, start: "1", uid: 1000, state: "S" } };
		return readProcessBirth(pid, boot, () => { throw Object.assign(new Error("fixture EACCES"), { code: "EACCES" }); });
	});
	const first = owner.stop(true);
	assert.equal(owner.stop(true), first);
	const result = await first;
	assert.equal(result.status, "unresolved");
	assert.deepEqual(result.unresolvedPids, [2147483647]);
});

test("unresolved cancellation returns while a descendant still holds stdout", { skip: process.platform !== "linux", timeout: 7000 }, async () => {
	const root = mkdtempSync(join(tmpdir(), "rosclaw-unresolved-pipe-"));
	const marker = join(root, "child.pid"), script = join(root, "parent.cjs");
	const childCode = `require('fs').writeFileSync(${JSON.stringify(marker)},String(process.pid));setInterval(()=>{},1000)`;
	writeFileSync(script, `require('child_process').spawn(process.execPath,['-e',${JSON.stringify(childCode)}],{detached:true,stdio:'inherit'}).unref();setInterval(()=>{},1000)`);
	const tool = buildWorkspacePackTools({ root, bwrapPath: () => null, bashProgressIntervalMs: 20 }).find((t) => t.name === "bash")!;
	const control = new AbortController();
	const originalRead = fs.readFileSync;
	let pid: number | undefined, start: string | undefined;
	let updates = 0;
	try {
		const pending = tool.execute("unresolved-pipe", { command: `${JSON.stringify(process.execPath)} ${JSON.stringify(script)}` }, control.signal, () => { updates++; }, {} as never);
		await until(() => existsSync(marker));
		pid = Number(readFileSync(marker, "utf8")); start = identity(pid)!.start;
		await delay(150); // Capture the descendant before injecting unreadability.
		fs.readFileSync = ((path: unknown, ...args: unknown[]) => {
			if (String(path) === `/proc/${pid}/stat`) throw Object.assign(new Error("fixture EACCES"), { code: "EACCES" });
			return (originalRead as (...values: unknown[]) => unknown)(path, ...args);
		}) as typeof fs.readFileSync;
		syncBuiltinESMExports();
		const began = performance.now(); control.abort();
		const result = await pending;
		assert.ok(performance.now() - began < 3000, "unresolved cleanup must not wait for pipe close");
		assert.equal(result.isError, true);
		assert.equal((result.details as { aborted: boolean }).aborted, true);
		assert.equal((result.details as { processCleanup: { status: string } }).processCleanup.status, "unresolved");
		assert.match(JSON.stringify(result.content), /cleanup unresolved.*may still be running/);
		const terminal = JSON.stringify(result), count = updates;
		fs.readFileSync = originalRead; syncBuiltinESMExports();
		assert.ok(identity(pid)?.running, "no unverified identity should have been signalled");
		cleanup(pid, start); // Release the inherited pipe only after the result.
		await delay(150);
		assert.equal(updates, count);
		assert.equal(JSON.stringify(result), terminal, "late close must not decode/resolve again");
	} finally {
		fs.readFileSync = originalRead; syncBuiltinESMExports(); control.abort();
		if (pid && start) cleanup(pid, start);
	}
});

test("unexpected identity-reader exceptions become unresolved, not rejected cleanup", { skip: process.platform !== "linux" }, async () => {
	let fail = false;
	const owner = new BashProcessOwnership(2147483647, (pid, boot) => {
		if (fail) throw new Error("unexpected fixture reader error");
		return { kind: "observed", birth: { pid, boot, ppid: 1, start: "1", uid: 1000, state: "S" } };
	});
	fail = true;
	assert.equal((await owner.stop(true)).status, "unresolved");
});
