/** PR-H9 红测试（结构防回归，总纲 v2 §18.1 删除清单 + 第 5 条 CI 扫描）。
 *
 * 红测试先行——删除前必须红：
 * 1. 旧 Worker UI 链文件不存在（completion-watch/job-widget/job-viewer/
 *    delegate/credentials migration/pi-worker-main/profiles/content-normalize）；
 * 2. extension/index.ts 不再 import 这些模块、不再注册 /jobs /job
 *    /workers /delegate 命令；
 * 3. main.ts 无 headless worker 入口；pi-runtime 无 migrateProviders；
 * 4. 本测试即"禁止重新引入"结构扫描——文件被重新创建即红。
 */
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

const SRC = join(import.meta.dirname, "../../src"); // dist/test → 包根/src

const DELETED_FILES = [
	"workers/completion-watch.ts",
	"workers/job-widget.ts",
	"workers/job-viewer.ts",
	"workers/pi-worker-main.ts",
	"workers/profiles.ts",
	"workers/content-normalize.ts",
	"tools/delegate.ts",
	"credentials/migration.ts",
];

test("H9: 旧 Worker UI 链文件已删除（禁止重新引入）", () => {
	const resurrected = DELETED_FILES.filter((f) => existsSync(join(SRC, f)));
	assert.deepEqual(resurrected, [], `旧内核文件被重新引入：${resurrected.join(", ")}`);
});

test("H9: extension 不再引用旧 Worker 链", () => {
	const source = readFileSync(join(SRC, "extension/index.ts"), "utf-8");
	for (const mod of [
		"completion-watch", "job-widget", "job-viewer",
		"WorkerCompletionWatcher", "JobsWidget", "JobViewerComponent",
	]) {
		assert.ok(!source.includes(mod), `extension 仍引用 ${mod}`);
	}
	for (const cmd of ['"jobs"', '"job"', '"workers"', '"delegate"']) {
		assert.ok(
			!source.includes(`registerCommand(${cmd}`),
			`extension 仍注册 /${cmd.replaceAll('"', "")} 命令`,
		);
	}
});

test("H9: headless worker 入口与凭据迁移已移除", () => {
	const main = readFileSync(join(SRC, "main.ts"), "utf-8");
	assert.ok(!main.includes("pi-worker-main"), "main.ts 仍有 headless worker 入口");
	assert.ok(!main.includes("runHeadlessWorker"), "main.ts 仍调 runHeadlessWorker");
	const runtime = readFileSync(join(SRC, "harness/pi/pi-runtime.ts"), "utf-8");
	assert.ok(!runtime.includes("migrateProviders"), "pi-runtime 仍做旧凭据迁移");
});
