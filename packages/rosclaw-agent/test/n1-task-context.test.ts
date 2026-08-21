/** PR-N1 红测试：ActiveTaskContext 唯一工作区（N 总纲 §5）。
 *
 * 红测试先行——修复前必须红：
 * 1. resolveTaskContext：显式 > cwd git 自动 > 持久化恢复 > default
 *    （~/.rosclaw/workspaces/default 真实创建）——带 provenance；
 * 2. 任务链禁止 process.cwd()（结构扫描：main/extension/product-pack/
 *    input-controller 不得再出现）；
 * 3. kernel bind_message 接受 workspace_root——任务 workspace_path =
 *    真实工作根（不再藏 ~/.rosclaw/tasks/<id>/workspace）；
 * 4. header/工具/登记同根：workspaceRoot 与 Bash pwd 永远一致。
 */
import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
	resolveTaskContext,
	type ActiveTaskContext,
} from "../src/native/active-task-context.js";
import { WorkspaceStore } from "../src/session/workspace.js";

function mkHome(): string {
	return mkdtempSync(join(tmpdir(), "n1-"));
}

test("N1: 显式 --workspace 优先", () => {
	const home = mkHome();
	const explicit = mkHome();
	const ctx = resolveTaskContext({
		rosclawHome: home, explicitWorkspace: explicit,
		cwd: mkHome(), mode: "SIMULATION",
	});
	assert.equal(ctx.workspaceRoot, explicit);
	assert.equal(ctx.workspaceSource, "explicit");
	assert.equal(ctx.artifactRoot, explicit);
});

test("N1: cwd 在 git 仓 → git root", () => {
	const home = mkHome();
	const repo = mkHome();
	mkdirSync(join(repo, ".git"));
	const nested = join(repo, "sub", "dir");
	mkdirSync(nested, { recursive: true });
	const ctx = resolveTaskContext({
		rosclawHome: home, cwd: nested, mode: "SIMULATION",
	});
	assert.equal(ctx.workspaceRoot, repo);
	assert.equal(ctx.workspaceSource, "git");
});

test("N1: 持久化 workspace 恢复（诚实标注 restored）", () => {
	const home = mkHome();
	const bound = mkHome();
	const store = new WorkspaceStore(home);
	store.bind(bound);
	const ctx = resolveTaskContext({
		rosclawHome: home, cwd: mkHome(), mode: "SIMULATION",
	});
	assert.equal(ctx.workspaceRoot, bound);
	assert.equal(ctx.workspaceSource, "restored");
});

test("N1: 什么都没有 → default workspace（真实创建，不谎称项目）", () => {
	const home = mkHome();
	const ctx = resolveTaskContext({
		rosclawHome: home, cwd: mkHome(), mode: "SIMULATION",
	});
	assert.equal(ctx.workspaceRoot, join(home, "workspaces", "default"));
	assert.equal(ctx.workspaceSource, "default");
	assert.ok(existsSync(ctx.workspaceRoot), "default workspace 必须真实创建");
});

test("N1: 上下文冻结且不可变", () => {
	const ctx = resolveTaskContext({
		rosclawHome: mkHome(), cwd: mkHome(), mode: "SIMULATION",
	});
	assert.ok(Object.isFrozen(ctx), "ActiveTaskContext 必须冻结");
	assert.equal(ctx.rosclawHome.length > 0, true);
	assert.ok(ctx.productRoot.length > 0, "productRoot 必须解析");
});

test("N1: 任务链禁止 process.cwd()（结构扫描）", async () => {
	const { readFileSync } = await import("node:fs");
	const offenders: string[] = [];
	for (const rel of [
		"src/main.ts",
		"src/extension/index.ts",
		"src/tools/product-pack.ts",
		"src/native/input-controller.ts",
	]) {
		const candidates = [
			join(import.meta.dirname, "..", rel),
			join(import.meta.dirname, "..", "..", rel),
		];
		const path = candidates.find((p) => existsSync(p));
		assert.ok(path, `找不到 ${rel}`);
		const source = readFileSync(path, "utf-8");
		// 方案 §5.3：除启动解析外禁止 process.cwd()——启动解析输入行
		// 必须带显式标注（"启动解析输入"），其余一律算违规。
		for (const line of source.split("\n")) {
			if (line.includes("process.cwd()") && !line.includes("启动解析输入")) {
				offenders.push(`${rel}: ${line.trim().slice(0, 60)}`);
			}
		}
	}
	assert.deepEqual(offenders, [], `任务链仍用 process.cwd()：${offenders.join(", ")}`);
});
