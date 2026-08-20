/** PR-H8 红测试（TS）：Task Activity/Logs/Artifacts 渲染 + 品牌扫描。
 *
 * 红测试先行——实现前必须红：
 * 1. renderTaskActivity：kernel 事件 → 中文阶段行（task.started、
 *    operation 系、artifact.created、verification.completed、
 *    task.terminal）；operation.output 不进 Activity（太吵——归 /logs）。
 * 2. renderOperationLogs：operation.output 尾部文本。
 * 3. renderArtifactList：产物名/路径/大小/哈希短码；空 → 诚实空行。
 * 4. 品牌扫描：extension 用户可见 description 不得出现 "Pi"（Gate A：
 *    用户 UI 中 Pi 品牌出现次数为 0）。
 */
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
	renderArtifactList,
	renderOperationLogs,
	renderTaskActivity,
	type KernelEvent,
} from "../src/native/task-activity.js";

function ev(seq: number, event_type: string, payload: Record<string, unknown>): KernelEvent {
	return { seq, event_type, payload };
}

test("H8: Activity 渲染阶段行（output 不进 Activity）", () => {
	const lines = renderTaskActivity([
		ev(1, "task.started", { goal: "画五角星" }),
		ev(2, "operation.started", { operation_id: "op_1", kind: "process", argv: ["python3", "sim.py"] }),
		ev(3, "operation.output", { text: "rendering...\n" }),
		ev(4, "operation.completed", { operation_id: "op_1", state: "SUCCEEDED", failure_code: "" }),
		ev(5, "artifact.created", { artifact_id: "art_1", path: "/ws/star.gif", bytes: 2048 }),
		ev(6, "verification.completed", { verification_id: "vrf_1", status: "PASS", checks: [{}, {}] }),
		ev(7, "task.terminal", { state: "SUCCEEDED", reason: "verification_passed" }),
	]);
	const joined = lines.join("\n");
	assert.match(joined, /任务开始/);
	assert.match(joined, /画五角星/);
	assert.match(joined, /进程/);
	assert.match(joined, /sim\.py/);
	assert.match(joined, /star\.gif/);
	assert.match(joined, /2048/);
	assert.match(joined, /验收/);
	assert.match(joined, /SUCCEEDED/);
	// operation.output 不进 Activity（归 /logs——否则刷屏）。
	assert.ok(!joined.includes("rendering..."), "output 不应进 Activity");
});

test("H8: 失败与修订也上 Activity", () => {
	const lines = renderTaskActivity([
		ev(1, "task.revised", { revision: 2, delta: "改成红色" }),
		ev(2, "operation.failed", { operation_id: "op_9", state: "FAILED", failure_code: "exit_1" }),
		ev(3, "verification.completed", { status: "FAIL", failures: ["缺少产物 star.gif"] }),
	]);
	const joined = lines.join("\n");
	assert.match(joined, /修订|r2/);
	assert.match(joined, /改成红色/);
	assert.match(joined, /exit_1/);
	assert.match(joined, /缺少产物/);
});

test("H8: Logs 渲染 operation.output 尾部", () => {
	const lines = renderOperationLogs([
		ev(1, "operation.started", { operation_id: "op_1", kind: "process", argv: ["python3", "sim.py"] }),
		ev(2, "operation.output", { text: "line-1\n" }),
		ev(3, "operation.output", { text: "line-2\n" }),
		ev(4, "operation.completed", { operation_id: "op_1", state: "SUCCEEDED", failure_code: "" }),
	]);
	const joined = lines.join("\n");
	assert.match(joined, /line-1/);
	assert.match(joined, /line-2/);
	assert.match(joined, /op_1/);
});

test("H8: Artifacts 列表与诚实空态", () => {
	const empty = renderArtifactList([]);
	assert.match(empty.join("\n"), /没有产物|无产物/);
	const lines = renderArtifactList([
		{ artifact_id: "art_abcdef123456", path: "/ws/star.gif", media_type: "image/gif", sha256: "deadbeefcafe", size_bytes: 2048 },
	]);
	const joined = lines.join("\n");
	assert.match(joined, /star\.gif/);
	assert.match(joined, /2048/);
	assert.match(joined, /deadbee/);
});

test("H8: 用户可见命令/快捷键 description 无 Pi 品牌（Gate A）", () => {
	const candidates = [
		join(import.meta.dirname, "../src/extension/index.ts"), // 源码直跑
		join(import.meta.dirname, "../../src/extension/index.ts"), // dist/test 编译后
	];
	const sourcePath = candidates.find((p) => existsSync(p));
	assert.ok(sourcePath, "找不到 extension/index.ts 源码");
	const source = readFileSync(sourcePath, "utf-8");
	const leaks: string[] = [];
	// registerCommand/registerShortcut 的 description 字面量。
	for (const m of source.matchAll(/description:\s*"([^"]*)"/g)) {
		if (/\bPi\b/.test(m[1])) leaks.push(m[1]);
	}
	assert.deepEqual(leaks, [], `用户可见 description 含 Pi 品牌：${leaks.join(" | ")}`);
});
