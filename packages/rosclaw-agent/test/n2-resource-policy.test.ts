/** PR-N2 红测试：恢复 Pi 的可信上下文与 Skill（N 总纲 §PR-N2）。
 *
 * 红测试先行——修复前必须红：
 * 1. ResourcePolicy 拆分 context/skill/extension/executable 四通道
 *    （旧布尔五连删）；
 * 2. developer：可信 AGENTS.md/CLAUDE.md 允许；任意项目 Skill/扩展/
 *    提示模板仍关；ROSClaw 内置签名 Skill 允许；
 * 3. robot：全关（不变）；
 * 4. 内置 Skill manifest digest 校验——篡改即排除 + 诊断；
 * 5. 上下文信任过滤：出根/超限文件被剔除且带 path+size 诊断。
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
	resourcePolicy,
	trustFilterContextFiles,
} from "../src/extension/resource-policy.js";
import { verifyBundledSkills } from "../src/extension/bundled-skills.js";

test("N2: developer 拆分策略——可信上下文允许 + 项目可执行仍关", () => {
	const p = resourcePolicy("developer");
	assert.equal(p.contextFiles, "trusted-readonly");
	assert.equal(p.skills, "bundled-signed");
	assert.equal(p.extensions, "off");
	assert.equal(p.promptTemplates, "off");
	assert.equal(p.executables, "off");
});

test("N2: robot 全关不变", () => {
	const p = resourcePolicy("robot");
	assert.equal(p.contextFiles, "off");
	assert.equal(p.skills, "off");
	assert.equal(p.extensions, "off");
	assert.equal(p.promptTemplates, "off");
	assert.equal(p.executables, "off");
});

test("N2: 内置 Skill digest 校验——篡改即排除", () => {
	const root = mkdtempSync(join(tmpdir(), "n2-skill-"));
	const skillDir = join(root, "skills", "rosclaw-embodied");
	mkdirSync(skillDir, { recursive: true });
	const content = "---\nname: rosclaw-embodied\ndescription: 具身任务纪律\n---\n\n# 具身\n";
	writeFileSync(join(skillDir, "SKILL.md"), content, "utf-8");
	const digest = createHash("sha256").update(content).digest("hex");
	writeFileSync(
		join(root, "skills", "manifest.json"),
		JSON.stringify({ skills: { "rosclaw-embodied": digest } }),
		"utf-8",
	);
	const ok = verifyBundledSkills(join(root, "skills"));
	assert.deepEqual(ok.verified, ["rosclaw-embodied"]);
	assert.deepEqual(ok.excluded, []);

	// 篡改后必须排除 + 诊断。
	writeFileSync(join(skillDir, "SKILL.md"), content + "\ntampered\n", "utf-8");
	const bad = verifyBundledSkills(join(root, "skills"));
	assert.deepEqual(bad.verified, []);
	assert.equal(bad.excluded.length, 1);
	assert.match(bad.excluded[0].reason, /digest|hash|篡改|digest 不符/i);
});

test("N2: 上下文信任过滤——出根与超限剔除", () => {
	const root = mkdtempSync(join(tmpdir(), "n2-ctx-"));
	const inside = join(root, "CLAUDE.md");
	writeFileSync(inside, "x".repeat(100), "utf-8");
	const outside = join(tmpdir(), "outside-CLAUDE.md");
	writeFileSync(outside, "evil", "utf-8");
	const result = trustFilterContextFiles(
		[
			{ path: inside, content: "x".repeat(100) },
			{ path: outside, content: "evil" },
			{ path: join(root, "AGENTS.md"), content: "y".repeat(200 * 1024) },
		],
		{ allowedRoots: [root], maxTotalBytes: 64 * 1024 },
	);
	assert.deepEqual(result.kept.map((f) => f.path), [inside]);
	assert.equal(result.diagnostics.length, 2);
	assert.ok(result.diagnostics.some((d) => d.includes(outside)));
	assert.ok(result.diagnostics.some((d) => /预算|budget|超/.test(d)));
});

// PR715: trust and read authority are independent. These tests exercise the
// production loader-options seam, not a substitute policy or a tool-call probe.
import { rmSync, symlinkSync } from "node:fs";
import {
	DefaultResourceLoader, ModelRuntime, loadSkillsFromDir, formatSkillsForPrompt,
} from "@earendil-works/pi-coding-agent";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import {
	bundledSkillLoaderOptions, createRosclawRuntime, type RosclawRuntimeOptions,
} from "../src/harness/pi/pi-runtime.js";
import { resolveTaskContext } from "../src/native/active-task-context.js";
import type { ToolCallBudget } from "../src/harness/pi/tool-call-budget.js";

function readPolicyFixture(t: { after(fn: () => void): void }) {
	const root = mkdtempSync(join(tmpdir(), "n2-read-policy-"));
	t.after(() => rmSync(root, { recursive: true, force: true }));
	const skills = join(root, "skills");
	const dir = join(skills, "rosclaw-embodied");
	mkdirSync(dir, { recursive: true });
	const file = join(dir, "SKILL.md");
	const content = "---\nname: rosclaw-embodied\ndescription: read policy fixture\n---\n\nTrusted fixture\n";
	writeFileSync(file, content);
	const manifest = join(skills, "manifest.json");
	writeFileSync(manifest, JSON.stringify({ skills: {
		"rosclaw-embodied": createHash("sha256").update(content).digest("hex"),
	} }));
	return { root, skills, dir, file, content, manifest };
}

function projectedSkills(options: ReturnType<typeof bundledSkillLoaderOptions>, file: string) {
	// Genuine PI1.3 Skill objects include required sourceInfo. Only the path is
	// varied to exercise an adversarial discovery result (e.g. OTHER.md).
	const discovered = loadSkillsFromDir({ dir: join(file, ".."), source: "path" });
	assert.equal(discovered.skills.length, 1);
	return options.skillsOverride({
		...discovered,
		skills: discovered.skills.map(skill => ({ ...skill, filePath: file })),
	});
}

test("PR715: read name alone cannot expose outside-W or exact-denied skills", t => {
	const f = readPolicyFixture(t);
	const workspace = join(f.root, "W");
	mkdirSync(workspace);
	const allowed = join(workspace, "allowed.md");
	writeFileSync(allowed, "allowed");
	const outside = bundledSkillLoaderOptions("developer", f.skills, workspace, {
		allowedTools: ["read"], exactPaths: { read: [allowed] },
	});
	const other = join(f.root, "other.md");
	writeFileSync(other, "other");
	const denied = bundledSkillLoaderOptions("developer", f.skills, f.root, {
		allowedTools: ["read"], exactPaths: { read: [other] },
	});
	for (const options of [outside, denied]) {
		assert.equal(options.noSkills, true);
		assert.deepEqual(options.additionalSkillPaths, []);
		const projected = projectedSkills(options, f.file);
		assert.deepEqual(projected.skills, []);
		assert.ok(projected.diagnostics.some(d => /READ_PATH_UNAUTHORIZED/.test(d.message)));
		assert.ok(projected.diagnostics.length <= 32);
	}
});

test("PR715: absent read never advertises a bash-readable skill", t => {
	const f = readPolicyFixture(t);
	for (const allowedTools of [[], ["bash"], ["edit"]]) {
		const options = bundledSkillLoaderOptions("developer", f.skills, f.root, { allowedTools });
		assert.deepEqual(options.additionalSkillPaths, []);
		assert.deepEqual(projectedSkills(options, f.file).skills, []);
	}
});

test("PR715: authorized trusted skill and no-policy developer default survive", t => {
	const f = readPolicyFixture(t);
	const budget: ToolCallBudget = {
		allowedTools: ["read"], exactPaths: { read: [f.file] },
		maxCalls: { read: 1 }, maxTotalCalls: 1,
	};
	const before = JSON.stringify(budget);
	for (const policy of [undefined, budget, { allowedTools: ["read"] },
		{ allowedTools: ["read"], exactPaths: {} }]) {
		const options = bundledSkillLoaderOptions("developer", f.skills, f.root, policy);
		assert.deepEqual(options.additionalSkillPaths, [f.dir]);
		const projected = projectedSkills(options, f.file);
		assert.deepEqual(projected.skills.map(s => s.filePath), [f.file]);
		assert.deepEqual(projected.diagnostics, []);
		// Even a loader result containing a different file cannot add authority.
		assert.deepEqual(projectedSkills(options, join(f.dir, "OTHER.md")).skills, []);
	}
	assert.equal(JSON.stringify(budget), before);
	const robot = bundledSkillLoaderOptions("robot", f.skills, f.root);
	assert.deepEqual(robot.additionalSkillPaths, []);
	assert.deepEqual(projectedSkills(robot, f.file).skills, []);
});

test("PR715: digest/manifest exclusions survive even with exact read authority", t => {
	for (const failure of ["tampered", "undeclared", "missing-manifest"]) {
		const f = readPolicyFixture(t);
		if (failure === "tampered") writeFileSync(f.file, f.content + "tampered");
		if (failure === "undeclared") writeFileSync(f.manifest, '{"skills":{}}');
		if (failure === "missing-manifest") rmSync(f.manifest);
		const options = bundledSkillLoaderOptions("developer", f.skills, f.root, {
			allowedTools: ["read"], exactPaths: { read: [f.file] },
		});
		assert.deepEqual(options.additionalSkillPaths, [], failure);
		const projected = projectedSkills(options, f.file);
		assert.deepEqual(projected.skills, [], failure);
		assert.ok(projected.diagnostics.length > 0, failure);
		assert.match(projected.diagnostics[0].message, /digest|manifest/);
	}
});

test("PR715: canonical exact aliases work, escaping/reentering aliases fail closed", t => {
	const f = readPolicyFixture(t);
	const alias = join(f.root, "alias");
	symlinkSync(f.dir, alias, "dir");
	const canonicalAlias = bundledSkillLoaderOptions("developer", f.skills, f.root, {
		allowedTools: ["read"], exactPaths: { read: [join(alias, "SKILL.md")] },
	});
	assert.deepEqual(canonicalAlias.additionalSkillPaths, [f.dir]);
	const outer = mkdtempSync(join(tmpdir(), "n2-outside-alias-"));
	t.after(() => rmSync(outer, { recursive: true, force: true }));
	// A path leaves W and reenters through a link. Final realpath containment
	// alone would accept this; budget checks every ancestor and must reject it.
	symlinkSync(f.dir, join(outer, "back"), "dir");
	symlinkSync(outer, join(f.root, "escape"), "dir");
	const sibling = join(f.root + "-sibling", "SKILL.md");
	for (const declared of [join(f.root, "escape", "back", "SKILL.md"),
		sibling, `${f.dir}/../SKILL.md`, f.dir]) {
		const options = bundledSkillLoaderOptions("developer", f.skills, f.root, {
			allowedTools: ["read"], exactPaths: { read: [declared] },
		});
		assert.deepEqual(options.additionalSkillPaths, [], declared);
		assert.deepEqual(projectedSkills(options, f.file).skills, [], declared);
	}
	// Also check the advertised skill path, not just declared ACL aliases.
	symlinkSync(f.skills, join(outer, "skills-back"), "dir");
	const escapedSkills = join(f.root, "escape", "skills-back");
	const escaped = bundledSkillLoaderOptions("developer", escapedSkills, f.root, {
		allowedTools: ["read"], exactPaths: { read: [f.file] },
	});
	assert.deepEqual(escaped.additionalSkillPaths, []);
	assert.deepEqual(projectedSkills(escaped, join(escapedSkills, "rosclaw-embodied", "SKILL.md")).skills, []);
	// A lexical workspace alias must bind identically to its canonical root.
	const workspaceAlias = join(outer, "workspace");
	symlinkSync(f.root, workspaceAlias, "dir");
	const options = bundledSkillLoaderOptions("developer", f.skills, workspaceAlias, {
		allowedTools: ["read"], exactPaths: { read: [f.file] },
	});
	assert.deepEqual(options.additionalSkillPaths, [f.dir]);
});

function deferred<T>() {
	let release: (value: T) => void = () => { throw new Error("DEFERRED_NOT_INITIALIZED"); };
	const promise = new Promise<T>(resolve => { release = resolve; });
	return { promise, release };
}

type Assembled = Awaited<ReturnType<typeof createRosclawRuntime>>;

function secondTrustedSkill(f: ReturnType<typeof readPolicyFixture>) {
	const dir = join(f.skills, "trusted-b");
	mkdirSync(dir);
	const file = join(dir, "SKILL.md");
	const content = "---\nname: trusted-b\ndescription: second trusted fixture\n---\n\nB\n";
	writeFileSync(file, content);
	writeFileSync(f.manifest, JSON.stringify({ skills: {
		"rosclaw-embodied": createHash("sha256").update(f.content).digest("hex"),
		"trusted-b": createHash("sha256").update(content).digest("hex"),
	} }));
	assert.deepEqual(verifyBundledSkills(f.skills).excluded, []);
	return { dir, file };
}

async function deferredAssembly(f: ReturnType<typeof readPolicyFixture>, budget?: ToolCallBudget) {
	const rosclawHome = join(f.root, "private-runtime");
	const options: RosclawRuntimeOptions = {
		cwd: f.root, rosclawHome, profile: "developer", version: "fixture",
		taskContext: { ...resolveTaskContext({ cwd: f.root, rosclawHome, mode: "SIMULATION" }), workspaceRoot: f.root },
		toolCallBudget: budget,
	};
	// No provider, stream, network, credential refresh, or process. This is only
	// the genuine SDK's empty offline model catalog, not a scientific backend.
	const modelRuntime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(),
		modelsPath: null, modelsStorePath: join(f.root, "catalog"),
		allowModelNetwork: false, refreshOnCreate: false });
	const gate = deferred<ModelRuntime>();
	const entered = deferred<void>();
	const pending = createRosclawRuntime(options, {
		bundledSkillsDir: f.skills,
		createModelRuntime: () => { entered.release(); return gate.promise; },
	});
	await entered.promise;
	return { options, pending, release: () => gate.release(modelRuntime) };
}

function assertLoaded(assembled: Assembled, files: string[]) {
	// Observe the actual production-created PI1.3 loader and actual session
	// prompt, not a fresh recomputation of the admission helper.
	const loader = assembled.runtime.services.resourceLoader;
	assert.ok(loader instanceof DefaultResourceLoader);
	assert.deepEqual(loader.getSkills().skills.map(skill => skill.filePath).sort(), [...files].sort());
	const prompt = assembled.runtime.session.systemPrompt;
	for (const file of files) assert.ok(prompt.includes(file), file);
	const rendered = formatSkillsForPrompt(loader.getSkills().skills, "read");
	if (files.length) assert.ok(prompt.includes(rendered.trim()));
	else assert.equal(rendered, "");
	return { loader, prompt };
}

async function assertRestrictive(assembled: Assembled, allowed: string, denied: string) {
	const runner = assembled.runtime.session.extensionRunner;
	assert.ok(runner);
	// Only denied public hook events: no file-tool execution and no consumed
	// calls. Zero caps make the original admission independently observable.
	assert.deepEqual(await runner.emitToolCall({ type: "tool_call", toolName: "read",
		toolCallId: "denied-path", input: { path: denied } }),
		{ block: true, reason: "TOOL_CALL_BUDGET_EXACT_PATH_REJECTED" });
	assert.deepEqual(await runner.emitToolCall({ type: "tool_call", toolName: "read",
		toolCallId: "original-cap", input: { path: allowed } }),
		{ block: true, reason: "TOOL_CALL_BUDGET_EXHAUSTED" });
	assert.deepEqual(await runner.emitToolCall({ type: "tool_call", toolName: "bash",
		toolCallId: "denied-name", input: { command: "never executed" } }),
		{ block: true, reason: "TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED" });
	assert.deepEqual(assembled.runtime.session.getActiveToolNames(), ["read"]);
}

for (const mutation of ["nested", "replacement", "undefined"] as const) {
	test(`PR715: real deferred factory retains admission after ${mutation} policy mutation`, async t => {
		const f = readPolicyFixture(t);
		const b = secondTrustedSkill(f);
		const policy = { allowedTools: ["read"], exactPaths: { read: [f.file] },
			maxCalls: { read: 0 }, maxTotalCalls: 0 };
		const boundary = await deferredAssembly(f, policy);
		let assembled: Assembled | undefined;
		const mutate = () => {
			if (mutation === "nested") {
				policy.allowedTools.push("bash");
				policy.exactPaths.read.splice(0, policy.exactPaths.read.length, b.file);
				policy.maxCalls.read = 100;
				policy.maxTotalCalls = 100;
			} else if (mutation === "replacement") {
				boundary.options.toolCallBudget = { allowedTools: ["read", "bash"],
					exactPaths: { read: [b.file] }, maxCalls: { read: 100 }, maxTotalCalls: 100 };
			} else boundary.options.toolCallBudget = undefined;
		};
		try {
			mutate(); // Deterministically suspended at the actual production await.
			boundary.release();
			assembled = await boundary.pending;
			const first = assertLoaded(assembled, [f.file]);
			assert.ok(!first.prompt.includes(b.file));
			await assertRestrictive(assembled, f.file, b.file);
			mutate();
			assert.deepEqual(await assembled.runtime.newSession(), { cancelled: false });
			const next = assertLoaded(assembled, [f.file]);
			assert.notEqual(next.loader, first.loader);
			assert.ok(!next.prompt.includes(b.file));
			await assertRestrictive(assembled, f.file, b.file);
		} finally {
			boundary.release();
			if (assembled) { await assembled.runtime.dispose(); assembled.ownership.releaseAll(); }
		}
	});
}

test("PR715: raw in-W alias retarget A to B cannot move canonical admission", async t => {
	const f = readPolicyFixture(t);
	const b = secondTrustedSkill(f);
	const alias = join(f.root, "declared-alias");
	symlinkSync(f.dir, alias, "dir");
	const boundary = await deferredAssembly(f, { allowedTools: ["read"],
		exactPaths: { read: [join(alias, "SKILL.md")] }, maxCalls: { read: 0 }, maxTotalCalls: 0 });
	let assembled: Assembled | undefined;
	try {
		rmSync(alias);
		symlinkSync(b.dir, alias, "dir");
		boundary.release();
		assembled = await boundary.pending;
		assertLoaded(assembled, [f.file]);
		await assertRestrictive(assembled, f.file, b.file);
		// Repeated factories must not rebind the raw alias, even after it returns
		// to A and then moves back to B between two real SDK replacements.
		for (const target of [f.dir, b.dir]) {
			rmSync(alias);
			symlinkSync(target, alias, "dir");
			assert.deepEqual(await assembled.runtime.newSession(), { cancelled: false });
			const observed = assertLoaded(assembled, [f.file]);
			assert.ok(!observed.prompt.includes(b.file));
			await assertRestrictive(assembled, f.file, b.file);
		}
	} finally {
		boundary.release();
		if (assembled) { await assembled.runtime.dispose(); assembled.ownership.releaseAll(); }
	}
});

for (const admission of ["no-policy", "no-read", "tampered", "untrusted"] as const) {
	test(`PR715: real factory preserves ${admission} behavior across replacements`, async t => {
		const f = readPolicyFixture(t);
		const policy = admission === "no-policy" ? undefined
			: { allowedTools: admission === "no-read" ? ["bash"] : ["read"], exactPaths: {} };
		const boundary = await deferredAssembly(f, policy);
		let assembled: Assembled | undefined;
		try {
			if (admission === "no-read") {
				assert.ok(policy);
				policy.allowedTools.push("read");
				boundary.options.toolCallBudget = undefined;
			}
			if (admission === "tampered") writeFileSync(f.file, f.content + "tampered");
			if (admission === "untrusted") writeFileSync(f.manifest, '{"skills":{}}');
			boundary.release();
			assembled = await boundary.pending;
			const expected = admission === "no-policy" ? [f.file] : [];
			const first = assertLoaded(assembled, expected);
			if (!expected.length) assert.ok(!first.prompt.includes(f.file));
			if (admission === "no-policy") {
				// Installing a later policy cannot retroactively replace admission.
				boundary.options.toolCallBudget = { allowedTools: [] };
			}
			assert.deepEqual(await assembled.runtime.newSession(), { cancelled: false });
			const next = assertLoaded(assembled, expected);
			assert.notEqual(next.loader, first.loader);
			if (!expected.length) assert.ok(!next.prompt.includes(f.file));
			if (admission === "tampered" || admission === "untrusted") {
				assert.ok(next.loader.getSkills().diagnostics.some(d => /digest|manifest/.test(d.message)));
			}
			// Trust must also be reverified after a previously accepted session.
			if (admission === "no-policy") {
				writeFileSync(f.file, f.content + "later tampering");
				assert.deepEqual(await assembled.runtime.newSession(), { cancelled: false });
				assertLoaded(assembled, []);
			}
		} finally {
			boundary.release();
			if (assembled) { await assembled.runtime.dispose(); assembled.ownership.releaseAll(); }
		}
	});
}
