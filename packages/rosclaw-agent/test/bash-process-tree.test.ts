import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(pred: () => boolean) {
	for (let i = 0; i < 200; i++) {
		if (pred()) return;
		await delay(10);
	}
	assert.fail("process fixture did not start");
}
function running(pid: number) {
	try {
		const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
		return !stat.slice(stat.lastIndexOf(")") + 2).startsWith("Z");
	} catch { return false; }
}
for (const kind of ["abort", "timeout"] as const) {
	test(`bash ${kind} terminates pipeline descendants and returns promptly`, { timeout: 6000 }, async () => {
		const root = mkdtempSync(join(tmpdir(), "rosclaw-tree-"));
		const pidPath = join(root, "pid");
		const bash = buildWorkspacePackTools({ root, bwrapPath: () => null }).find((t) => t.name === "bash")!;
		const control = new AbortController();
		const start = Date.now();
		const result = bash.execute("tree", {
			command: `sh -c 'echo $$ > ${pidPath}; trap "" TERM; sleep 60' | cat`,
			...(kind === "timeout" ? { timeout_sec: 0.4 } : {}),
		}, control.signal, undefined, {} as never);
		await until(() => existsSync(pidPath) && Number(readFileSync(pidPath, "utf8")) > 0);
		const pid = Number(readFileSync(pidPath, "utf8"));
		assert.ok(running(pid));
		if (kind === "abort") control.abort();
		const out = await result;
		assert.ok(Date.now() - start < 3500);
		assert.match(JSON.stringify(out.content), kind === "abort" ? /ABORTED/ : /TIMEOUT/);
		await until(() => !running(pid));
	});
}
test("silent bash emits progress and removes heartbeat after completion", async () => {
	const root = mkdtempSync(join(tmpdir(), "rosclaw-progress-"));
	const bash = buildWorkspacePackTools({ root, bwrapPath: () => null, bashProgressIntervalMs: 20 }).find((t) => t.name === "bash")!;
	const updates: unknown[] = [];
	await bash.execute("progress", { command: "sleep 0.15" }, undefined, (u) => { updates.push(u); }, {} as never);
	assert.ok(updates.length > 0);
	const count = updates.length;
	await delay(70);
	assert.equal(updates.length, count);
});
