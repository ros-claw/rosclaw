import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

type Update = { content: { text?: string }[]; details?: Record<string, unknown> };
const text = (u: Update) => u.content.map((c) => c.text ?? "").join("\n");

test("bash heartbeat shows fresh output after the final-output cap fills", async () => {
	const root = mkdtempSync(join(tmpdir(), "rosclaw-live-tail-"));
	const bash = buildWorkspacePackTools({ root, bwrapPath: () => null, bashProgressIntervalMs: 20 })
		.find((tool) => tool.name === "bash")!;
	const updates: Update[] = [];
	const result = await bash.execute("large-progress", {
		command: "node -e \"process.stdout.write('x'.repeat(70000)); setTimeout(() => process.stderr.write('LATEST_PROGRESS_MARKER'), 60); setTimeout(() => {}, 200)\"",
	}, undefined, (u) => { updates.push(u as Update); }, {} as never);
	assert.ok(updates.some((u) => text(u).includes("LATEST_PROGRESS_MARKER")),
		"live progress must retain new stderr after stdout filled the final-result cap");
	assert.ok(updates.every((u) => text(u).length < 5000), "heartbeat tail remains bounded");
	assert.match(text(result as Update), /exit=0 wall=/);
	assert.ok(!text(result as Update).includes("LATEST_PROGRESS_MARKER"),
		"the existing first-output truncation contract must not change");
});

test("bash heartbeat distinguishes output silence from the configured deadline", async () => {
	const root = mkdtempSync(join(tmpdir(), "rosclaw-quiet-clock-"));
	const bash = buildWorkspacePackTools({ root, bwrapPath: () => null, bashProgressIntervalMs: 20 })
		.find((tool) => tool.name === "bash")!;
	const updates: Update[] = [];
	const result = await bash.execute("quiet-progress", {
		command: "node -e \"process.stdout.write('INITIAL_OUTPUT'); setTimeout(() => {}, 200)\"",
		timeout_sec: 1,
	}, undefined, (u) => { updates.push(u as Update); }, {} as never);
	assert.ok(updates.some((u) => text(u).includes("INITIAL_OUTPUT")
		&& Number(u.details?.outputQuietMs) >= 50));
	assert.ok(updates.every((u) => typeof u.details?.elapsedMs === "number"
		&& Number(u.details?.outputQuietMs) <= Number(u.details?.elapsedMs)
		&& Number(u.details?.timeoutRemainingMs) >= 0
		&& Number(u.details?.timeoutRemainingMs) <= 1000));
	assert.match(text(updates.at(-1)!), /output quiet=.*s; timeout remaining=.*s/);
	assert.match(text(result as Update), /exit=0 wall=/);
});
