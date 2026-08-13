/** 十一审 PR-D 红测试：Workspace 一等状态。
 *
 * 1. gitRootOf：repo 内子目录归一到 root；非 git 返回 null；
 * 2. WorkspaceStore：bind/recent/persist/reload；
 * 3. resolveStartupWorkspace：显式 > cwd git 自动 > 既有绑定；
 * 4. delegate 未指定 workspace 时默认注入绑定值；
 * 5. header 显示 Project 名（无绑定显示 —）。
 */

import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

async function makeStore() {
	const { WorkspaceStore } = await import("../src/session/workspace.js");
	const home = mkdtempSync(join(tmpdir(), "rh-ws-"));
	return new WorkspaceStore(home);
}

function makeRepo(): string {
	const repo = mkdtempSync(join(tmpdir(), "repo-"));
	mkdirSync(join(repo, ".git"));
	mkdirSync(join(repo, "src/deep"), { recursive: true });
	writeFileSync(join(repo, "README.md"), "x");
	return repo;
}

test("gitRootOf 归一 + bind/recent/持久化", async () => {
	const repo = makeRepo();
	const { gitRootOf } = await import("../src/session/workspace.js");
	assert.equal(gitRootOf(join(repo, "src/deep")), repo);
	assert.equal(gitRootOf("/tmp"), null);
	const store = await makeStore();
	const bound = store.bind(join(repo, "src"));
	assert.equal(bound, repo); // git 归一
	assert.equal(store.current, repo);
	assert.deepEqual(store.recent, [repo]);
	// reload 持久化
	const { WorkspaceStore } = await import("../src/session/workspace.js");
	const home = mkdtempSync(join(tmpdir(), "rh-ws-"));
	const s1 = new WorkspaceStore(home);
	s1.bind(repo);
	const s2 = new WorkspaceStore(home);
	assert.equal(s2.current, repo);
});

test("resolveStartupWorkspace：显式 > cwd git 自动 > 既有", async () => {
	const repo = makeRepo();
	const { resolveStartupWorkspace } = await import("../src/session/workspace.js");
	const store = await makeStore();
	// 显式
	const explicit = resolveStartupWorkspace(store, repo, "/tmp");
	assert.equal(explicit.bound, repo);
	assert.equal(explicit.auto, false);
	// cwd git 自动
	const store2 = await makeStore();
	const auto = resolveStartupWorkspace(store2, undefined, join(repo, "src/deep"));
	assert.equal(auto.bound, repo);
	assert.equal(auto.auto, true);
	// 无 git cwd → 用既有绑定
	const plain = mkdtempSync(join(tmpdir(), "plain-"));
	const store3 = await makeStore();
	store3.bind(repo);
	const keep = resolveStartupWorkspace(store3, undefined, plain);
	assert.equal(keep.bound, repo);
	assert.equal(keep.auto, false);
	// 无 git 无绑定 → null
	const store4 = await makeStore();
	const none = resolveStartupWorkspace(store4, undefined, plain);
	assert.equal(none.bound, null);
});

test("delegate 默认注入绑定 workspace", async () => {
	const { ActiveSessionContext } = await import("../src/session/active-context.js");
	const { buildDelegateTool } = await import("../src/tools/delegate.js");
	const calls: Array<Record<string, unknown>> = [];
	const active = new ActiveSessionContext({
		sessionId: "pi_test",
		missionId: "mis_1",
		contextRevision: 1,
		mode: "SIMULATION",
		profile: "developer",
		contextState: "FRESH",
		leaseState: "ACTIVE",
		actionsAllowed: true,
	});
	const center = {
		call: async (_method: string, params?: unknown) => {
			calls.push((params as { request?: { arguments?: Record<string, unknown> } })?.request?.arguments ?? {});
			return { ok: true, result: { ok: true, status: "STARTED", summary: "WorkOrder: wo_x" } };
		},
	};
	const tool = buildDelegateTool({
		rosclawHome: "/tmp/rh",
		active,
		center,
		workspace: () => "/home/user/myrepo",
	} as never);
	await tool.execute("t1", { goal: "改代码" }, undefined, undefined, {} as never);
	assert.equal(calls[0].workspace, "/home/user/myrepo");
	// 模型显式指定时不覆盖。
	await tool.execute("t2", { goal: "x", workspace: "/other" }, undefined, undefined, {} as never);
	assert.equal(calls[1].workspace, "/other");
});

test("header 显示 Project 名（无绑定显示 —）", async () => {
	const { renderHeader } = await import("../src/ui/product-state.js");
	const base = {
		snapshot_seq: 1,
		product_version: "1.2.0",
		kernel: "READY",
		model: "K3",
		mode: "SIMULATION",
		context_state: "FRESH",
		context_revision: 1,
		lease_state: "ACTIVE",
		operator: "OFFLINE",
		action_readiness: { state: "READY", reason_codes: [], snapshot_seq: 1 },
		mission_id: "mis_1",
	};
	const withWs = renderHeader({ ...base, workspace: "rosclaw" } as never, "zh-CN");
	assert.match(withWs, /Project rosclaw/);
	const without = renderHeader(base as never, "zh-CN");
	assert.match(without, /Project —/);
});
